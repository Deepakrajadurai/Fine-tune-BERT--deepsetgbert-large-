import json

log_path = r"C:\Users\vijayakr\.gemini\antigravity-ide\brain\03965544-5803-45ac-ac4c-7592c74de654\.system_generated\logs\transcript.jsonl"

print("Searching transcript for run_command tool calls...")
with open(log_path, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f):
        try:
            data = json.loads(line)
            if "tool_calls" in data:
                for tc in data["tool_calls"]:
                    name = tc.get("name")
                    args = tc.get("args", {})
                    if name == "run_command":
                        cmd = args.get("CommandLine", "")
                        print(f"Line {line_num}: {cmd}")
        except Exception as e:
            pass
