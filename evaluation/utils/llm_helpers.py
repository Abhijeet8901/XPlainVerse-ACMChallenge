import json
import os
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_MODEL_CACHE = {}
_BERT_SCORER_CACHE = {}
_SLE_CACHE = {}


def load_text(path):
    return Path(path).read_text()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def list_json_files(input_dir):
    return sorted(
        path
        for path in Path(input_dir).iterdir()
        if path.is_file() and path.suffix == ".json"
    )


def render_prompt(template_text, replacements):
    prompt = template_text
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt


def maybe_limit(items, limit):
    if limit is None:
        return items
    return items[:limit]


def get_api_key(explicit_key):
    if explicit_key:
        return explicit_key
    return os.environ.get("OPENAI_API_KEY", "EMPTY")


def get_runtime_device():
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_torch_dtype(torch_dtype_name):
    if torch_dtype_name in (None, "auto"):
        return "auto"

    import torch

    if not hasattr(torch, torch_dtype_name):
        raise ValueError("Unsupported torch dtype: {0}".format(torch_dtype_name))
    return getattr(torch, torch_dtype_name)


def _extract_message_text(message):
    content = message.get("content", "")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)

    return str(content)


def chat_completion(
    backend,
    model,
    system_prompt,
    user_prompt,
    base_url=None,
    api_key=None,
    temperature=0.0,
    max_tokens=1200,
    timeout=300,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=False,
    attn_implementation=None,
    cache_dir=None,
    enable_thinking=False,
):
    if backend == "transformers":
        return transformers_chat_completion(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            cache_dir=cache_dir,
            enable_thinking=enable_thinking,
        )

    if backend != "openai_compatible":
        raise ValueError("Unsupported inference backend: {0}".format(backend))

    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + get_api_key(api_key),
    }
    request = Request(url, data=body, headers=headers, method="POST")

    try:
        with urlopen(request, timeout=timeout) as response:
            response_text = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            "Chat completion request failed with HTTP status {0}: {1}".format(
                exc.code, detail
            )
        )
    except URLError as exc:
        raise RuntimeError(
            "Failed to reach the chat completion endpoint at {0}: {1}".format(
                url, exc
            )
        )

    payload = json.loads(response_text)
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("No choices returned by the chat completion endpoint.")

    message = choices[0].get("message") or {}
    return _extract_message_text(message).strip()


def chat_completion_batch(
    backend,
    model,
    system_prompt,
    user_prompts,
    base_url=None,
    api_key=None,
    temperature=0.0,
    max_tokens=1200,
    timeout=300,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=False,
    attn_implementation=None,
    cache_dir=None,
    enable_thinking=False,
):
    prompts = list(user_prompts)
    if not prompts:
        return []

    if backend == "transformers":
        return transformers_chat_completion_batch(
            model=model,
            system_prompt=system_prompt,
            user_prompts=prompts,
            temperature=temperature,
            max_tokens=max_tokens,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            cache_dir=cache_dir,
            enable_thinking=enable_thinking,
        )

    return [
        chat_completion(
            backend=backend,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            cache_dir=cache_dir,
            enable_thinking=enable_thinking,
        )
        for user_prompt in prompts
    ]


def preload_chat_model(
    backend,
    model,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=False,
    attn_implementation=None,
    cache_dir=None,
):
    if backend != "transformers":
        return None
    return _load_transformers_model(
        model_name=model,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
        cache_dir=cache_dir,
    )


def clear_chat_model_cache():
    _MODEL_CACHE.clear()
    try:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _get_bert_score_model_with_safetensors(model_type, num_layers, all_layers=None):
    import torch
    import bert_score.utils as bert_score_utils
    from transformers import AutoModel

    if model_type.startswith("scibert"):
        model = AutoModel.from_pretrained(
            bert_score_utils.cache_scibert(model_type),
            use_safetensors=True,
        )
    elif "t5" in model_type:
        from transformers import T5EncoderModel

        model = T5EncoderModel.from_pretrained(
            model_type,
            use_safetensors=True,
        )
    else:
        model = AutoModel.from_pretrained(
            model_type,
            use_safetensors=True,
        )
    model.eval()

    if hasattr(model, "decoder") and hasattr(model, "encoder"):
        model = model.encoder

    if not all_layers:
        if hasattr(model, "n_layers"):
            assert 0 <= num_layers <= model.n_layers
            model.n_layers = num_layers
        elif hasattr(model, "layer"):
            assert 0 <= num_layers <= len(model.layer)
            model.layer = torch.nn.ModuleList(
                [layer for layer in model.layer[:num_layers]]
            )
        elif hasattr(model, "encoder"):
            if hasattr(model.encoder, "albert_layer_groups"):
                assert 0 <= num_layers <= model.encoder.config.num_hidden_layers
                model.encoder.config.num_hidden_layers = num_layers
            elif hasattr(model.encoder, "block"):
                assert 0 <= num_layers <= len(model.encoder.block)
                model.encoder.block = torch.nn.ModuleList(
                    [layer for layer in model.encoder.block[:num_layers]]
                )
            else:
                assert 0 <= num_layers <= len(model.encoder.layer)
                model.encoder.layer = torch.nn.ModuleList(
                    [layer for layer in model.encoder.layer[:num_layers]]
                )
        elif hasattr(model, "transformer"):
            assert 0 <= num_layers <= len(model.transformer.layer)
            model.transformer.layer = torch.nn.ModuleList(
                [layer for layer in model.transformer.layer[:num_layers]]
            )
        elif hasattr(model, "layers"):
            assert 0 <= num_layers <= len(model.layers)
            model.layers = torch.nn.ModuleList(
                [layer for layer in model.layers[:num_layers]]
            )
        else:
            raise ValueError("Not supported")
    else:
        if hasattr(model, "output_hidden_states"):
            model.output_hidden_states = True
        elif hasattr(model, "encoder"):
            model.encoder.output_hidden_states = True
        elif hasattr(model, "transformer"):
            model.transformer.output_hidden_states = True

    return model


def _load_transformers_model(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=False,
    attn_implementation=None,
    cache_dir=None,
):
    cache_key = (
        model_name,
        device_map,
        str(torch_dtype),
        trust_remote_code,
        attn_implementation,
        str(cache_dir),
    )
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    normalized_device_map = device_map
    if isinstance(normalized_device_map, str):
        stripped_value = normalized_device_map.strip()
        if stripped_value.lower() in {"", "none", "null"}:
            normalized_device_map = None
        else:
            normalized_device_map = stripped_value

    manual_device = None
    if isinstance(normalized_device_map, str) and (
        normalized_device_map == "cpu"
        or normalized_device_map == "cuda"
        or normalized_device_map.startswith("cuda:")
    ):
        manual_device = normalized_device_map

    model_kwargs = {
        "torch_dtype": _resolve_torch_dtype(torch_dtype),
        "trust_remote_code": trust_remote_code,
    }
    if manual_device is None and normalized_device_map is not None:
        model_kwargs["device_map"] = normalized_device_map
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    if cache_dir:
        model_kwargs["cache_dir"] = cache_dir

    tokenizer_kwargs = {"trust_remote_code": trust_remote_code}
    if cache_dir:
        tokenizer_kwargs["cache_dir"] = cache_dir

    try:
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if manual_device is not None:
            model = model.to(torch.device(manual_device))
        model.eval()
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        loaded = {
            "loader_type": "causal_lm",
            "processor": tokenizer,
            "model": model,
        }
        _MODEL_CACHE[cache_key] = loaded
        return loaded
    except Exception as causal_lm_error:
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError:
            raise causal_lm_error

        processor_kwargs = {"trust_remote_code": trust_remote_code}
        if cache_dir:
            processor_kwargs["cache_dir"] = cache_dir

        processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
        model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
        if manual_device is not None:
            import torch

            model = model.to(torch.device(manual_device))
        model.eval()
        loaded = {
            "loader_type": "image_text_to_text",
            "processor": processor,
            "model": model,
        }
        _MODEL_CACHE[cache_key] = loaded
        return loaded


def _build_prompt_from_messages(system_prompt, user_prompt):
    return "System:\n{0}\n\nUser:\n{1}\n\nAssistant:\n".format(
        system_prompt.strip(), user_prompt.strip()
    )


def _build_causal_lm_prompt_texts(processor, system_prompt, user_prompts, enable_thinking):
    prompt_texts = []
    for user_prompt in user_prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(processor, "apply_chat_template"):
            prompt_texts.append(
                processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            )
        else:
            prompt_texts.append(_build_prompt_from_messages(system_prompt, user_prompt))
    return prompt_texts


def _extract_input_ids_and_attention_mask(prompt_inputs):
    if hasattr(prompt_inputs, "items"):
        input_ids = prompt_inputs["input_ids"]
        attention_mask = prompt_inputs.get("attention_mask")
    else:
        input_ids = prompt_inputs
        attention_mask = None
    return input_ids, attention_mask


def _left_pad_tokenized_inputs(tokenized_inputs, pad_token_id):
    import torch

    max_length = max(input_ids.shape[-1] for input_ids, _ in tokenized_inputs)
    padded_input_ids = []
    padded_attention_masks = []

    for input_ids, attention_mask in tokenized_inputs:
        seq_length = input_ids.shape[-1]
        pad_length = max_length - seq_length
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if pad_length > 0:
            input_ids = torch.nn.functional.pad(input_ids, (pad_length, 0), value=pad_token_id)
            attention_mask = torch.nn.functional.pad(attention_mask, (pad_length, 0), value=0)
        padded_input_ids.append(input_ids)
        padded_attention_masks.append(attention_mask)

    return {
        "input_ids": torch.cat(padded_input_ids, dim=0),
        "attention_mask": torch.cat(padded_attention_masks, dim=0),
    }


def _build_image_text_prompt_texts(processor, system_prompt, user_prompts, enable_thinking):
    prompt_texts = []
    for user_prompt in user_prompts:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]
        if hasattr(processor, "apply_chat_template"):
            prompt_texts.append(
                processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            )
        else:
            prompt_texts.append(_build_prompt_from_messages(system_prompt, user_prompt))
    return prompt_texts


def _move_to_device(batch_like, device):
    if hasattr(batch_like, "items"):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch_like.items()
        }
    if hasattr(batch_like, "to"):
        return batch_like.to(device)
    return batch_like


def transformers_chat_completion(
    model,
    system_prompt,
    user_prompt,
    temperature=0.0,
    max_tokens=1200,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=False,
    attn_implementation=None,
    cache_dir=None,
    enable_thinking=False,
):
    import torch

    loaded = _load_transformers_model(
        model_name=model,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
        cache_dir=cache_dir,
    )
    processor = loaded["processor"]
    loaded_model = loaded["model"]
    loader_type = loaded["loader_type"]

    if loader_type == "causal_lm":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(processor, "apply_chat_template"):
            prompt_inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=enable_thinking,
            )
        else:
            prompt_inputs = processor(
                _build_prompt_from_messages(system_prompt, user_prompt),
                return_tensors="pt",
            )["input_ids"]

        prompt_inputs = _move_to_device(prompt_inputs, loaded_model.device)
        if hasattr(prompt_inputs, "items"):
            input_ids = prompt_inputs["input_ids"]
            attention_mask = prompt_inputs.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            generation_kwargs = dict(prompt_inputs)
            generation_kwargs["attention_mask"] = attention_mask
        else:
            input_ids = prompt_inputs
            attention_mask = torch.ones_like(input_ids)
            generation_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        generation_kwargs.update({
            "max_new_tokens": max_tokens,
            "pad_token_id": processor.pad_token_id,
            "eos_token_id": processor.eos_token_id,
            "use_cache": True,
            "num_beams": 1,
        })
        if temperature and temperature > 0.0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
        else:
            generation_kwargs["do_sample"] = False

        with torch.inference_mode():
            output_ids = loaded_model.generate(**generation_kwargs)

        generated_ids = output_ids[0][input_ids.shape[-1] :]
        return processor.decode(generated_ids, skip_special_tokens=True).strip()

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]
    if hasattr(processor, "apply_chat_template"):
        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    else:
        prompt_text = _build_prompt_from_messages(system_prompt, user_prompt)

    model_inputs = processor(text=prompt_text, return_tensors="pt")
    model_inputs = _move_to_device(model_inputs, loaded_model.device)
    generation_kwargs = {
        "max_new_tokens": max_tokens,
        "use_cache": True,
        "num_beams": 1,
    }
    if temperature and temperature > 0.0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
    else:
        generation_kwargs["do_sample"] = False

    with torch.inference_mode():
        output_ids = loaded_model.generate(**model_inputs, **generation_kwargs)

    input_ids = model_inputs.get("input_ids")
    input_length = input_ids.shape[-1] if input_ids is not None else 0
    generated_ids = output_ids[0][input_length:]
    if hasattr(processor, "decode"):
        return processor.decode(generated_ids, skip_special_tokens=True).strip()
    return processor.batch_decode([generated_ids], skip_special_tokens=True)[0].strip()


def transformers_chat_completion_batch(
    model,
    system_prompt,
    user_prompts,
    temperature=0.0,
    max_tokens=1200,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=False,
    attn_implementation=None,
    cache_dir=None,
    enable_thinking=False,
):
    import torch

    prompts = list(user_prompts)
    if not prompts:
        return []

    loaded = _load_transformers_model(
        model_name=model,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
        cache_dir=cache_dir,
    )
    processor = loaded["processor"]
    loaded_model = loaded["model"]
    loader_type = loaded["loader_type"]

    if loader_type == "causal_lm":
        tokenized_inputs = []
        for user_prompt in prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            if hasattr(processor, "apply_chat_template"):
                prompt_inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    enable_thinking=enable_thinking,
                )
            else:
                prompt_inputs = processor(
                    _build_prompt_from_messages(system_prompt, user_prompt),
                    return_tensors="pt",
                )["input_ids"]
            tokenized_inputs.append(_extract_input_ids_and_attention_mask(prompt_inputs))

        model_inputs = _left_pad_tokenized_inputs(
            tokenized_inputs,
            processor.pad_token_id,
        )
        model_inputs = _move_to_device(model_inputs, loaded_model.device)
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": processor.pad_token_id,
            "eos_token_id": processor.eos_token_id,
            "use_cache": True,
            "do_sample": bool(temperature and temperature > 0.0),
            "num_beams": 1,
        }
        if temperature and temperature > 0.0:
            generation_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output_ids = loaded_model.generate(**model_inputs, **generation_kwargs)

        input_length = model_inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_length:]
        return [
            text.strip()
            for text in processor.batch_decode(generated_ids, skip_special_tokens=True)
        ]

    prompt_texts = _build_image_text_prompt_texts(
        processor,
        system_prompt,
        prompts,
        enable_thinking,
    )
    model_inputs = processor(text=prompt_texts, padding=True, return_tensors="pt")
    model_inputs = _move_to_device(model_inputs, loaded_model.device)
    generation_kwargs = {
        "max_new_tokens": max_tokens,
        "use_cache": True,
        "do_sample": bool(temperature and temperature > 0.0),
        "num_beams": 1,
    }
    if temperature and temperature > 0.0:
        generation_kwargs["temperature"] = temperature

    with torch.inference_mode():
        output_ids = loaded_model.generate(**model_inputs, **generation_kwargs)

    input_length = model_inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_length:]
    if hasattr(processor, "batch_decode"):
        return [text.strip() for text in processor.batch_decode(generated_ids, skip_special_tokens=True)]
    return [processor.decode(ids, skip_special_tokens=True).strip() for ids in generated_ids]


def extract_first_json(text):
    text = text.strip()
    if not text:
        raise ValueError("Empty model response.")

    candidates = []
    for fenced_match in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE):
        candidates.append(fenced_match.group(1).strip())
    candidates.append(text)

    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except ValueError:
            pass

        best_match = None
        best_dict_match = None
        for index, char in enumerate(candidate):
            if char not in "[{":
                continue
            try:
                obj, end_index = decoder.raw_decode(candidate[index:])
            except ValueError:
                continue
            span_length = end_index
            if best_match is None or span_length > best_match[0]:
                best_match = (span_length, obj)
            if isinstance(obj, dict) and (
                best_dict_match is None or span_length > best_dict_match[0]
            ):
                best_dict_match = (span_length, obj)
        if best_dict_match is not None:
            return best_dict_match[1]
        if best_match is not None:
            return best_match[1]

    raise ValueError("Could not parse a JSON object from the model response.")


def _normalize_tokenizer_model_max_length(tokenizer, model):
    config = getattr(model, "config", None)
    max_positions = None
    for attr_name in (
        "max_position_embeddings",
        "n_positions",
        "max_seq_len",
        "max_sequence_length",
    ):
        value = getattr(config, attr_name, None)
        if isinstance(value, int) and value > 0:
            max_positions = value
            break

    if max_positions is None:
        return

    tokenizer_max_length = getattr(tokenizer, "model_max_length", None)
    if not isinstance(tokenizer_max_length, int) or tokenizer_max_length <= 0 or tokenizer_max_length > 1000000:
        normalized_length = int(max_positions)
    else:
        normalized_length = int(min(tokenizer_max_length, max_positions))

    tokenizer.model_max_length = normalized_length
    if hasattr(tokenizer, "init_kwargs") and isinstance(tokenizer.init_kwargs, dict):
        tokenizer.init_kwargs["model_max_length"] = normalized_length


def get_bert_scorer(
    model_type,
    lang="en",
    rescale_with_baseline=False,
    device=None,
):
    if device is None:
        device = get_runtime_device()

    cache_key = (model_type, lang, bool(rescale_with_baseline), device)
    if cache_key in _BERT_SCORER_CACHE:
        return _BERT_SCORER_CACHE[cache_key]

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    from bert_score import BERTScorer
    import bert_score.scorer as bert_score_scorer
    import bert_score.utils as bert_score_utils

    original_scorer_get_model = bert_score_scorer.get_model
    original_utils_get_model = bert_score_utils.get_model
    bert_score_scorer.get_model = _get_bert_score_model_with_safetensors
    bert_score_utils.get_model = _get_bert_score_model_with_safetensors
    try:
        scorer = BERTScorer(
            model_type=model_type,
            lang=lang,
            rescale_with_baseline=rescale_with_baseline,
            device=device,
        )
    finally:
        bert_score_scorer.get_model = original_scorer_get_model
        bert_score_utils.get_model = original_utils_get_model
    _normalize_tokenizer_model_max_length(scorer._tokenizer, scorer._model)
    _BERT_SCORER_CACHE[cache_key] = scorer
    return scorer


def preload_bertscorer(
    model_type,
    lang="en",
    rescale_with_baseline=False,
    device=None,
):
    return get_bert_scorer(
        model_type=model_type,
        lang=lang,
        rescale_with_baseline=rescale_with_baseline,
        device=device,
    )


def get_sle_components(model_id, local_files_only=False, device=None):
    if device is None:
        device = get_runtime_device()

    cache_key = (model_id, bool(local_files_only), device)
    if cache_key in _SLE_CACHE:
        return _SLE_CACHE[cache_key]

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch_device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        )
    except ValueError as exc:
        error_text = str(exc)
        if "Due to a serious vulnerability issue in `torch.load`" in error_text:
            raise RuntimeError(
                "Failed to load SLE model '{0}'. This Hugging Face repo currently depends on "
                "a legacy '.bin' checkpoint rather than 'model.safetensors', and your current "
                "Torch version is below 2.6 so Transformers blocks loading it. Upgrade Torch "
                "to >= 2.6 to use this SLE model."
                .format(model_id)
            ) from exc
        raise
    model.to(torch_device)
    model.eval()

    loaded = {
        "tokenizer": tokenizer,
        "model": model,
        "device": torch_device,
    }
    _SLE_CACHE[cache_key] = loaded
    return loaded


def preload_sle_model(model_id, local_files_only=False, device=None):
    return get_sle_components(
        model_id=model_id,
        local_files_only=local_files_only,
        device=device,
    )


def build_progress_bar(iterable, *, total=None, desc=None, disable=False):
    if disable:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)
    except Exception:
        return iterable


def get_reference_entities(payload):
    if not isinstance(payload, dict):
        raise ValueError(
            "Expected reference extraction JSON object but got {0}. "
            "This usually means the model returned a top-level list or "
            "the outer JSON object was truncated and only an inner list parsed."
            .format(type(payload).__name__)
        )
    return payload.get("diagnostic_entities", payload.get("entities", []))


def get_reference_claims(payload):
    if not isinstance(payload, dict):
        raise ValueError(
            "Expected reference extraction JSON object but got {0}. "
            "This usually means the model returned a top-level list or "
            "the outer JSON object was truncated and only an inner list parsed."
            .format(type(payload).__name__)
        )
    return payload.get("evidence_claims", payload.get("atomic_facts", []))


def get_coverage_entity_matches(payload):
    if not isinstance(payload, dict):
        raise ValueError(
            "Expected coverage JSON object but got {0}."
            .format(type(payload).__name__)
        )
    return payload.get("entity_matches", payload.get("entity_results", []))


def get_coverage_claim_matches(payload):
    if not isinstance(payload, dict):
        raise ValueError(
            "Expected coverage JSON object but got {0}."
            .format(type(payload).__name__)
        )
    return payload.get(
        "claim_matches",
        payload.get("claim_results", payload.get("fact_results", [])),
    )


def compute_coverage_summary(entity_results, claim_results):
    entity_total = len(entity_results)
    claim_total = len(claim_results)
    entity_present = sum(1 for item in entity_results if item.get("present"))
    claim_present = sum(1 for item in claim_results if item.get("present"))
    summary = {
        "entity_total": entity_total,
        "entity_present": entity_present,
        "entity_coverage": float(entity_present) / entity_total if entity_total else 0.0,
        "claim_total": claim_total,
        "claim_present": claim_present,
        "claim_coverage": float(claim_present) / claim_total if claim_total else 0.0,
    }
    summary["fact_total"] = summary["claim_total"]
    summary["fact_present"] = summary["claim_present"]
    summary["fact_coverage"] = summary["claim_coverage"]
    return summary
