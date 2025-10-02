from openai import OpenAI
from time import sleep
from openai import RateLimitError
import os

class GPTPlayer():
    def __init__(self, api_key=""):
        if api_key == "":
            self.api_key = os.getenv('OPENAI_API_KEY')
        else:
            self.api_key = api_key
        self.completion_tokens = 0
        self.prompt_tokens = 0

    def get_LLM_action(self, system_prompt, user_prompt, model='gpt-4o',
                       temperature=0.7, json_format=False, seed=None,
                       stop=[], max_tokens=200, actions=None) -> str:
        from openai import OpenAI
        # from openai import RateLimitError  # ensure this is imported somewhere
        client = OpenAI(api_key=self.api_key)

        def _is_valid_json(txt: str) -> bool:
            try:
                import json
                json.loads(txt)
                return True
            except Exception:
                return False

        def _salvage_json(txt: str) -> str:
            # Return the largest balanced {...} substring if possible
            try:
                import json
                json.loads(txt)
                return txt
            except Exception:
                pass
            start = txt.find("{")
            end = txt.rfind("}")
            if start != -1 and end != -1 and end > start:
                cand = txt[start:end + 1]
                try:
                    import json
                    json.loads(cand)
                    return cand
                except Exception:
                    return txt
            return txt

        def _call_api(max_toks: int):
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                stream=False,
                max_tokens=max_toks,
            )
            if json_format:
                kwargs["response_format"] = {"type": "json_object"}
            # Only pass stop if provided (empty list can behave oddly)
            if stop:
                kwargs["stop"] = stop
            # if seed is not None: kwargs["seed"] = seed  # enable if your SDK supports it
            return client.chat.completions.create(**kwargs)

        try:
            response = _call_api(max_tokens)
        except RateLimitError:
            from time import sleep
            sleep(5)
            print('rate limit error')
            return self.get_LLM_action(system_prompt, user_prompt, model, temperature, json_format, seed, stop,
                                       max_tokens, actions)

        choice = response.choices[0]
        outputs = choice.message.content or ""
        finish_reason = getattr(choice, "finish_reason", None)

        # log completion tokens
        self.completion_tokens += getattr(response.usage, "completion_tokens", 0)
        self.prompt_tokens += getattr(response.usage, "prompt_tokens", 0)

        # If JSON was requested, make sure we didn't get cut off
        if json_format:
            needs_retry = (finish_reason == "length") or (not _is_valid_json(outputs))
            if needs_retry:
                # single retry with larger budget
                try:
                    more_tokens = max(1024, int(max_tokens * 2))
                    response2 = _call_api(more_tokens)
                    choice2 = response2.choices[0]
                    outputs2 = choice2.message.content or outputs  # fall back to original if empty
                    self.completion_tokens += getattr(response2.usage, "completion_tokens", 0)
                    self.prompt_tokens += getattr(response2.usage, "prompt_tokens", 0)
                    outputs = outputs2
                except Exception:
                    pass

            # If still not valid JSON, salvage best-effort substring
            if not _is_valid_json(outputs):
                outputs = _salvage_json(outputs)

            return outputs, True

        return outputs, False

    def get_LLM_query(self, system_prompt, user_prompt, temperature=0.7, model='gpt-4o', json_format=False, seed=None, stop=[], max_tokens=200):
        client = OpenAI(api_key=self.api_key)
        # client = AzureOpenAI()
        try:
            output_padding = ''
            if json_format:
                output_padding  = '\n{"'
                
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt+output_padding}
                ],
                temperature=temperature,
                stream=False,
                stop=stop,
                max_tokens=max_tokens
            )
            message = response.choices[0].message.content
        except RateLimitError:
            # sleep 5 seconds and try again
            sleep(5)  
            print('rate limit error1')
            return self.get_LLM_query(system_prompt, user_prompt, temperature, model, json_format, seed, stop, max_tokens)
        
        if json_format:
            json_start = 0
            json_end = message.find('}') + 1 # find the first "}
            message_json = '{"' + message[json_start:json_end]
            if len(message_json) > 0:
                return message_json, True
        return message, False
