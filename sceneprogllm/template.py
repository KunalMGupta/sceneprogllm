from uuid import uuid4
from string import Template

class SceneProgTemplate:
    def __init__(self, template_str=None):
        self.name = "Template_" + str(uuid4())
        self.template_str = template_str
        self.template = Template(template_str)

    def to_string(self):
        """Returns the raw template string with placeholders."""
        return self.template.template

    def substitute(self, **kwargs):
        """Returns a string with provided placeholders safely substituted."""
        return self.template.safe_substitute(**kwargs)

    def get_section(self, start_marker, end_marker):
        """Extracts a section from the raw template string using start and end markers."""
        start = self.template_str.find(start_marker)
        if start == -1:
            raise ValueError(f"Start marker '{start_marker}' not found!")
        start += len(start_marker)

        end = self.template_str.find(end_marker, start)
        if end == -1:
            raise ValueError(f"End marker '{end_marker}' not found!")

        return self.template_str[start:end].strip()

    @staticmethod
    def format(template_str, placeholders):
        """Static method to safely substitute values into a string template."""
        assert isinstance(template_str, str), "template_str must be a string"
        return Template(template_str).safe_substitute(placeholders)