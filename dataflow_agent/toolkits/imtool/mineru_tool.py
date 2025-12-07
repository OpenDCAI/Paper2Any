# Before Using, do these:

# hf download opendatalab/MinerU2.5-2509-1.2B --local-dir opendatalab/MinerU2.5-2509-1.2B

# With vllm>=0.10.1, you can use following command to serve the model. The logits processor is used to support no_repeat_ngram_size sampling param, which can help the model to avoid generating repeated content.

# vllm serve opendatalab/MinerU2.5-2509-1.2B --host 127.0.0.1 --port <port> \
#   --logits-processors mineru_vl_utils:MinerULogitsProcessor
# If you are using vllm<0.10.1, no_repeat_ngram_size sampling param is not supported. You still can serve the model without logits processor:

# vllm serve opendatalab/MinerU2.5-2509-1.2B --host 127.0.0.1 --port <port>


from PIL import Image
from mineru_vl_utils import MinerUClient


# ---------------------------------------
# 1. two_step_extract (sync)
# ---------------------------------------
def run_two_step_extract(image_path: str, port: int):
    image = Image.open(image_path)
    client = MinerUClient(
        backend="http-client",
        server_url=f"http://127.0.0.1:{port}"
    )
    return client.two_step_extract(image)


# ---------------------------------------
# 2. batch_two_step_extract (sync)
# ---------------------------------------
def run_batch_two_step_extract(image_paths: list[str], port: int):
    images = [Image.open(p) for p in image_paths]
    client = MinerUClient(
        backend="http-client",
        server_url=f"http://127.0.0.1:{port}"
    )
    return client.batch_two_step_extract(images)


# ---------------------------------------
# 3. aio_two_step_extract (async)
# ---------------------------------------
async def run_aio_two_step_extract(image_path: str, port: int):
    image = Image.open(image_path)
    client = MinerUClient(
        backend="http-client",
        server_url=f"http://127.0.0.1:{port}"
    )
    return await client.aio_two_step_extract(image)


# ---------------------------------------
# 4. aio_batch_two_step_extract (async)
# ---------------------------------------
async def run_aio_batch_two_step_extract(image_paths: list[str], port: int):
    images = [Image.open(p) for p in image_paths]
    client = MinerUClient(
        backend="http-client",
        server_url=f"http://127.0.0.1:{port}"
    )
    return await client.aio_batch_two_step_extract(images)
