from pathlib import Path
import json
import os
from utils.gpt5_utils import call_model_wo_tools


def load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

SYSTEM_PROMPT = load_prompt("prompts/clarifier_system_prompt.txt")
USER_PROMPT = load_prompt("prompts/clarifier_prompt.txt")

class Clarifier:
    def __init__(self,config):
        self.schema_file = config.get("schema_file_path")
        self.max_keys = config.get("max_key_concpets",5)

    def task_spec(self, user_query):

        if os.path.exists(os.path.join(self.schema_file, "template.json")):
            task_schema_file = os.path.join(self.schema_file, "template.json")
        else:    
            raise FileNotFoundError(f"No template.json found in {self.schema_file}")

        with open(task_schema_file, 'r', encoding="utf-8") as f:
            schema = json.load(f)


        system_prompt = SYSTEM_PROMPT

        user_prompt = USER_PROMPT.format(
            user_query=user_query,
            schema_json=json.dumps(schema, indent=2),
            max_keys=self.max_keys,
        )
        
        response = call_model_wo_tools(system_prompt=system_prompt, user_prompt=user_prompt)

        return response

    def _parse_result(self, result):
        """Parse the LLM response into structured format"""
        try:
            # Try to extract JSON from the response
            if "{" in result and "}" in result:
                start_idx = result.find("{")
                end_idx = result.rfind("}") + 1
                json_str = result[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return {"raw_output": result, "error": "No structured output found"}
        except Exception as e:
            return {
                "raw_output": result, 
                "error": f"Failed to parse response: {str(e)}"
            }

    def run(self, raw_input):
        result = self.task_spec(raw_input)
        contract = self._parse_result(result)
        return contract
