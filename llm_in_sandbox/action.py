import re
from typing import Dict
import shlex


class Action:
    """
    Represents an action with:
      - function_name (e.g. 'file_editor')
      - parameters    (a dictionary of parameter_name -> value)
    """

    def __init__(self, function_name: str, parameters: Dict[str, str], function_id: str = None):
        self.function_name = function_name
        self.parameters = parameters

    @classmethod
    def from_string(cls, action_str: str) -> "Action":
        """
        Parses a string of the form:
          <function=FUNCTION_NAME>
            <parameter=KEY>VALUE</parameter>
            ...
          </function>
        """
        fn_match = re.search(r"<function\s*=\s*([^>]+)>", action_str)
        function_name = fn_match.group(1).strip() if fn_match else ""

        pattern = r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>"
        param_matches = re.findall(pattern, action_str, flags=re.DOTALL)

        params = {}
        for param_key, param_value in param_matches:
            params[param_key.strip()] = param_value.strip()

        return cls(function_name, params)

    def __str__(self) -> str:
        return self.to_xml_string()

    def to_xml_string(self) -> str:
        xml_str = f"<function={self.function_name}>\n"
        for param_key, param_value in self.parameters.items():
            xml_str += f"  <parameter={param_key}>{param_value}</parameter>\n"
        xml_str += "</function>"
        return xml_str

    def to_dict(self) -> Dict[str, object]:
        return {"function": self.function_name, "parameters": self.parameters}

    def to_bashcmd(self) -> str:
        if not self.function_name:
            return ""
        elif self.function_name == "finish" or self.function_name == "submit":
            return "echo '<<<Finished>>>'"

        cmd_parts = [shlex.quote(self.function_name)]
        base_command = self.parameters.get("command")
        if base_command is not None:
            cmd_parts.append(shlex.quote(base_command))

        for param_key, param_value in self.parameters.items():
            if param_key == "command":
                continue
            param_value_quoted = shlex.quote(str(param_value))
            cmd_parts.append(f"--{param_key}")
            cmd_parts.append(param_value_quoted)

        return " ".join(cmd_parts)
