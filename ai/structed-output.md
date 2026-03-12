## Usages

### Router

```python
def main():
    router = Router()
    instruction = "Debug issue and fix it and send mail"  # Example instruction
    instruction = "Send  me mail about progress of issue"  # Example instruction
    route_decision = router.route(instruction)
    print(f"Route Decision: {route_decision}")

from enum import Enum
from pydantic import BaseModel

import instructor
import os

class RouteType(str, Enum):
    KNOWLEDGE = "knowledge"     # Knowledge → RAG/Agent
    CODE = "code"               # Code → Search/Diff
    DEBUG = "debug"             # Debug → Agent multi-step
    ACTION_READ = "action_read"    # Action-Read → Direct execution
    ACTION_WRITE = "action_write"  # Action-Write → Requires confirmation

class RouteDecision(BaseModel):
    route: RouteType
    risk_level: str   # low / medium / high
    need_confirm: bool
    confirm_message: str | None  # "Are you sure you want to send an email to XXX?"

class Router:
    def __init__(self):
        pass

    def route(self, instruction) -> RouteDecision:
        client = instructor.from_provider("openai/{model_name}".format(model_name=os.getenv("MODEL_NAME")))
        return client.chat.completions.create(
            response_model=RouteDecision,
            messages=[{"role": "user", "content": instruction}],
        )
```

## References

- <https://github.com/567-labs/instructor>