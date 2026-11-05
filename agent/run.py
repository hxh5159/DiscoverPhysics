"""Editable LLM discovery loop shared by public and organizer-hosted episodes.

Run with ``python -m agent.run --model 27b``. Model startup/readiness
precedes mission(), so it is separate from the 600-second exploration window.
No simulator, validation reference, answer library or external agent framework
is imported by this agent. Its model proposes experiments and actual law.py.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import re
import signal
import time

from rift import Console
from rift.law import validate_source
from scripts.validate_claims import validate_claims
from .tools import candidate_tool
from .models import InferenceError


SYSTEM_PROMPT = """你是自动物理规律发现智能体。在未知二维环境中通过允许的观测建立可预测探针轨迹的规律。
任务名不告诉你真实规律。你只能使用任务包和合法实验，不读取任何环境实现或隐藏验证。
每次只返回一个JSON对象，无Markdown。动作：
{"action":"observe","times":[0,0.5,1]}
{"action":"launch","t_launch":0,"readout":[0,0.1,0.5],"source_properties":{},"probes":[{"pos":[x,y],"vel":[vx,vy],"properties":{}}]}
{"action":"view_record","index":0,"start":0,"count":10}
{"action":"archive"}
{"action":"check","law":"完整Python源代码"}
{"action":"fit","law":"完整Python源代码","max_nfev":12}
{"action":"draft","law":"完整Python源代码","claims":{"dynamics_order":2,"hidden_matter":false,"time_dependence":false}}
{"action":"submit","law":"完整Python源代码","claims":{"dynamics_order":2,"hidden_matter":false,"time_dependence":false}}
所有坐标、属性和时刻按mission公开目录选择，不能把示例数字当作合法范围。source_properties与每个probe.properties须包含该宇宙目录的所有键，默认值见目录。物理时刻必须对齐0.005；t_launch为绝对物理时刻，readout为相对偏移。探针互不作用且不影响背景，但会自由运动。
初始first_light包含合法探针轨迹和可见背景；每次新探针收费1，observe免费但有独立次数；位置和速度有观测噪声。根据轨迹、方向、属性和投放相位辨别规律，不要直接差分噪声位置后过拟合。实验设计、推理、拟合均消耗本局现实时间。请尽早保存有效draft再改进；submit结束本局。
law.py须包含import numpy as np、LAW_API_VERSION='1.0.0'、PARAMS={至多5个有限标量}及def acceleration(t,bodies,params)。可用math/numpy/scipy纯数学；禁止IO、随机、联网。元数据必须字面量。可选BODY_PROPS按可见ID提供未知标量属性，可选HIDDEN列出候选隐藏对象初态，但不增加claims字段。
每个body是只读字典，键id、kind(background/hidden/probe)、pos二维数组、vel二维数组、properties标量字典。加速度须返回(len(bodies),2)有限实数数组。回调可能仅含背景/候选隐藏对象，也可能额外含一枚探针。背景的真实作用角色未知，不能默认所有背景都被固定，或默认它们均有相同源强。你负责返回这些对象的候选加速度；官方积分器统一推进背景再投放探针。
check/fit工具仅在已有观测上调用固定积分器；不会查询真实模拟器或隐藏测试。fit只优化你在PARAMS中提出的标量，不能替你提出结构。工具残差是带噪观测拟合误差，不是正式分数。若你把未知常数写死在函数中，fit不能优化它。
普通结果在模型上下文中可能抽稀到10个时刻，但完整数据仍可用于工具。view_record按start/count分页读取已付费完整记录，count最多10，依次增加start可检查所有时刻；分页不会增加实验或改变观测数据。
claims三项由你判断：dynamics_order作用决定速度填1、决定加速度填2；hidden_matter表示位置不可见的作用源是否存在（不是可见天体源强未知）；time_dependence表示固定几何和属性后规律是否显式随物理时间变。两布尔值严格true/false。示例字段值只是格式，不能当正确答案。
最后必须提供可执行law.py与claims，不提交JSON数学表达式或预先计算轨迹。剩余时间少时优先提交最佳已有候选。"""


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def parse_action(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Model returned no final action; reduce thinking length or increase output allowance.")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I).strip()
    decoder = json.JSONDecoder()
    try:
        answer = json.loads(text)
    except ValueError:
        # Some local model templates put a brief sentence before their JSON.
        start = text.find('{')
        if start < 0:
            raise ValueError("Return exactly one JSON action object.") from None
        answer, end = decoder.raw_decode(text[start:])
        if text[start+end:].strip():
            raise ValueError("Only one JSON action is accepted.")
    if not isinstance(answer, dict) or not isinstance(answer.get("action"), str):
        raise ValueError("Action must be an object containing action.")
    return answer


def compact(value, *, points=10):
    """Thin only model-context copies; full legal observations remain on disk."""
    if isinstance(value, dict):
        answer = {k: compact(v, points=points) for k,v in value.items()}
        times = value.get("times")
        if isinstance(times, list) and len(times) > points:
            indices = sorted({round(i*(len(times)-1)/(points-1)) for i in range(points)})
            for key in ("times", "readout", "positions", "velocities", "visible_background_positions"):
                items = value.get(key)
                if isinstance(items, list) and len(items) == len(times):
                    answer[key] = [items[i] for i in indices]
            answer["context_sample_indices"] = indices
            answer["full_time_count"] = len(times)
        return answer
    if isinstance(value, list):
        return [compact(v, points=points) for v in value]
    return value


class DiscoveryAgent:
    """Dependency-injected loop: tests may supply a mock, normal CLI never does."""

    def __init__(self, console, model, output_dir, *, max_tokens=4096,
                 request_timeout=300., tool_timeout=30., max_turns=3,
                 max_context_chars=28000, clock=time.monotonic):
        self.console, self.model = console, model
        self.directory = Path(output_dir).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_tokens, self.request_timeout = int(max_tokens), float(request_timeout)
        self.tool_timeout, self.max_turns = float(tool_timeout), int(max_turns)
        self.max_context_chars, self.clock = int(max_context_chars), clock
        self.mission = None
        self.records, self.events, self.history = [], [], []
        self.latest = None
        self.deadline = None
        self.last_budget = None

    def _save_json(self, name, value):
        (self.directory/name).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding="utf-8")

    def remaining(self):
        return max(0., self.deadline-self.clock())

    def _record(self, record):
        if isinstance(record, dict) and all(k in record for k in ("probes", "positions", "readout", "t_launch")):
            detached = json.loads(_json(record))
            detached.setdefault("source_properties", {})
            self.records.append(detached)
            self._save_json("observations.json", self.records)
        if isinstance(record, dict) and isinstance(record.get("budget"), dict):
            self.last_budget = record["budget"]

    def _pair(self, source, claims, *, submit=False):
        if not isinstance(source, str):
            raise ValueError("law must contain complete Python source text.")
        validate_source(source)
        validate_claims(claims)
        if self.remaining() <= 0:
            raise TimeoutError("Exploration ended before candidate preparation.")
        # Separate filenames ensure an invalid revision never replaces our
        # last locally accepted pair. The service also checks atomically.
        number = len(self.events)+1
        law = self.directory/f"candidate-{number:04d}.py"
        claim_file = self.directory/f"candidate-{number:04d}.claims.json"
        law.write_text(source, encoding="utf-8")
        claim_file.write_text(_json(claims), encoding="utf-8")
        if self.remaining() <= 0:
            raise TimeoutError("Exploration ended before artifact submission.")
        method = self.console.submit if submit else self.console.save_draft
        result = method(law=str(law), claims=str(claim_file))
        self.latest = {"law": source, "claims": claims}
        (self.directory/"law.py").write_text(source, encoding="utf-8")
        self._save_json("claims.json", claims)
        return result

    def _action(self, action):
        if self.remaining() <= 0:
            raise TimeoutError("Exploration window ended.")
        name = action["action"]
        if name == "observe":
            result = self.console.observe(action["times"])
        elif name == "launch":
            result = self.console.launch(t_launch=action["t_launch"], readout=action["readout"],
                probes=action["probes"], source_properties=action.get("source_properties", {}))
            self._record(result)
        elif name == "archive":
            result = self.console.archive()
        elif name == "view_record":
            index = action["index"]
            if type(index) is not int or not 0 <= index < len(self.records):
                raise ValueError("Unknown observation index.")
            record = self.records[index]
            total = len(record["times"])
            start, count = action.get("start", 0), action.get("count", 10)
            if (type(start) is not int or type(count) is not int or not 0 <= start < total
                    or not 1 <= count <= 10):
                raise ValueError("view_record requires a valid start index and count from 1 through 10.")
            indices = list(range(start, min(start+count, total)))
            page = dict(record)
            for key in ("times", "readout", "positions", "velocities", "visible_background_positions"):
                if isinstance(record.get(key), list) and len(record[key]) == total:
                    page[key] = [record[key][i] for i in indices]
            return {"index": index, "full_time_count": total, "sample_indices": indices, "record": page}
        elif name in ("check", "fit"):
            result = candidate_tool(action["law"], self.mission, self.records,
                fit=name == "fit", max_nfev=action.get("max_nfev", 12),
                timeout=max(.001, min(self.tool_timeout, self.remaining())))
            if self.remaining() <= 0:
                raise TimeoutError("Exploration ended while running the candidate tool.")
        elif name in ("draft", "submit"):
            result = self._pair(action["law"], action["claims"], submit=name == "submit")
        else:
            raise ValueError("Unknown action; use observe/launch/archive/view_record/check/fit/draft/submit.")
        if isinstance(result, dict) and isinstance(result.get("budget"), dict):
            self.last_budget = result["budget"]
        return result

    def _messages(self, turn):
        state = {"remaining_seconds": round(self.remaining(), 3), "budget": self.last_budget,
                 "record_count": len(self.records), "saved_candidate": self.latest,
                 "turn": turn, "max_turns": self.max_turns,
                 "remaining_model_calls_including_current":
                     self.max_turns-turn+1 if self.max_turns else None,
                 "instruction": "Use view_record index/start/count pages to inspect full data; submit a valid candidate before time expires."}
        if self.max_turns:
            state["instruction"] += " Plan experiments and candidate generation within the remaining model calls."
            if turn == self.max_turns:
                state["instruction"] += " This is the final model call. Return submit with complete law.py and claims now; reuse the best saved candidate if needed."
        essential = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "MISSION:\n"+_json(compact(self.mission))},
            {"role": "user", "content": "CURRENT_STATE:\n"+_json(state)}]
        retained, used = [], sum(len(x["content"]) for x in essential)
        for message in reversed(self.history):
            length = len(message["content"])
            if used+length > self.max_context_chars:
                break
            retained.append(message)
            used += length
        return essential+list(reversed(retained))

    def run(self):
        begin = self.clock()
        self.mission = self.console.mission()
        allowance = float(self.mission["limits"]["wall_clock_seconds"])
        # Conservatively count the mission request itself; no network delay
        # can accidentally grant more time than the server session.
        self.deadline = min(begin+allowance, self.clock()+float(
            self.mission["budget"]["wall_clock_remaining_seconds"]))
        self.last_budget = self.mission["budget"]
        self._save_json("mission.json", self.mission)
        self._record(self.mission["first_light"])
        status, turn = "wall_clock_expired", 0
        try:
            while self.remaining() > 0 and (self.max_turns == 0 or turn < self.max_turns):
                turn += 1
                if self.remaining() < 3 and self.latest:
                    self._pair(self.latest["law"], self.latest["claims"], submit=True)
                    status = "submitted"
                    break
                row = {"turn": turn, "remaining_before": self.remaining()}
                try:
                    response = self.model.complete(self._messages(turn), max_tokens=self.max_tokens,
                        timeout=max(.001, min(self.request_timeout, self.remaining())))
                    if self.remaining() <= 0:
                        row["late_response_rejected"] = True
                        self.events.append(row)
                        break
                    action = parse_action(response)
                    row["action"] = action
                    result = self._action(action)
                    row["result"] = result
                    self.history.extend([{"role": "assistant", "content": response},
                        {"role": "user", "content": "TOOL_RESULT:\n"+_json(compact(result))}])
                    if action["action"] == "submit":
                        status = "submitted"
                except InferenceError as exc:
                    row["inference_error"] = str(exc)[:1000]
                    self.events.append(row)
                    status = "inference_error"
                    raise
                except TimeoutError as exc:
                    row["error"] = str(exc)[:1000]
                    self.events.append(row)
                    break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {str(exc)[:1200]}"
                    row["error"] = error
                    self.history.append({"role": "user", "content": "ACTION_ERROR: "+error})
                self.events.append(row)
                self._save_json("events.json", self.events)
                if status == "submitted":
                    break
            if status != "submitted" and self.remaining() > 0 and self.latest:
                self._pair(self.latest["law"], self.latest["claims"], submit=True)
                status = "submitted_after_optional_turn_limit"
            elif self.max_turns and turn >= self.max_turns and self.remaining() > 0:
                status = "optional_turn_limit_no_submission"
        finally:
            self._save_json("events.json", self.events)
            summary = {"status": status, "turns": turn, "max_turns": self.max_turns,
                "elapsed_seconds": self.clock()-begin,
                "has_saved_pair": self.latest is not None,
                "model_transport": getattr(self.model, "capability_label", "injected_transport_not_model_verification"),
                "capability_limit": "Successful execution does not establish prediction quality."}
            self._save_json("agent_summary.json", summary)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-file", default=os.environ.get("SHORE_MODEL_REGISTRY"),
                        help="Model registry supplied by the organizer or used for public development.")
    parser.add_argument("--model-ref", help="Registry model id; omitted selects primary_model_id.")
    parser.add_argument("--model", default=os.environ.get("SHORE_MODEL", "27b"))
    parser.add_argument("--base-url", default=os.environ.get("SHORE_MODEL_BASE_URL"))
    parser.add_argument("--model-id", default=os.environ.get("SHORE_MODEL_ID"))
    parser.add_argument("--model-root", default=os.environ.get("SHORE_MODEL_ROOT"))
    parser.add_argument("--model-path", default=os.environ.get("SHORE_MODEL_PATH"))
    parser.add_argument("--model-python", default=os.environ.get("SHORE_MODEL_PYTHON"))
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--startup-timeout", type=float, default=900.)
    parser.add_argument("--output-dir", default=os.environ.get("RIFT_OUTPUT_DIR", "agent-output"))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--request-timeout", type=float, default=300.)
    parser.add_argument("--tool-timeout", type=float, default=30.)
    parser.add_argument("--max-turns", type=int, default=3,
                        help="Maximum model calls per universe (default 3); zero uses only the session time budget.")
    args = parser.parse_args(argv)
    if (min(args.max_tokens, args.tp_size) <= 0 or args.max_turns < 0
            or any(not math.isfinite(v) or v <= 0 for v in
                (args.startup_timeout, args.request_timeout, args.tool_timeout))):
        parser.error("Limits must be finite positive values; max-turns may be zero.")
    model_key, model_path, adapter_path = args.model, args.model_path, None
    if args.models_file:
        from rift.strategy import load_model_registry, model_binding
        try:
            binding = model_binding(load_model_registry(args.models_file, check_files=True), args.model_ref)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        model_key, model_path = binding['base_model'], binding['path']
        adapter_path = binding.get('adapter_path')
    elif args.model_ref:
        parser.error('--model-ref requires --models-file or SHORE_MODEL_REGISTRY.')
    from .models import managed_model
    # Models are loaded and a real model list is checked BEFORE mission.
    previous_term = signal.getsignal(signal.SIGTERM)
    def stop_requested(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, stop_requested)
    try:
        with managed_model(model_key, base_url=args.base_url, model_id=args.model_id,
                model_root=args.model_root, model_path=model_path, adapter_path=adapter_path,
                python_executable=args.model_python, tensor_parallel_size=args.tp_size,
                startup_timeout=args.startup_timeout, log_dir=Path(args.output_dir)/"model-logs") as model:
            summary = DiscoveryAgent(Console(), model, args.output_dir,
                max_tokens=args.max_tokens, request_timeout=args.request_timeout,
                tool_timeout=args.tool_timeout, max_turns=args.max_turns).run()
    except InferenceError as exc:
        summary = {"status": "model_service_error", "message": str(exc)[:1000]}
        directory = Path(args.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory/"startup_or_service_error.json").write_text(_json(summary), encoding="utf-8")
        print(_json(summary))
        return 2
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["has_saved_pair"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
