# Hugging Face Space image for the read-only KB Arena demo.
# It installs the published wheel, so the Space never builds this repository.
FROM python:3.12-slim

# Spaces run the container as uid 1000, so the demo files belong to that user.
RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH

RUN pip install --no-cache-dir --user kb-arena==0.11.0

# Seed the results from the wheel. The aws-compute glob is the whole allow list.
# The datasets directory stays empty: no corpus file enters this image, so no
# reader can pull one out of it.
RUN mkdir -p /home/user/demo/results /home/user/demo/datasets \
    && cp "$(python -c 'import kb_arena, pathlib; print(pathlib.Path(kb_arena.__file__).parent / "data")')"/aws-compute_*.json \
       /home/user/demo/results/

# An operator sets demo mode here on purpose. That published state is what lets
# the read gate serve the demo results without a token, and it answers 503 on
# chat, arena, and tool routes. Add no model key and no API token to this Space.
ENV KB_ARENA_DEMO_MODE=true \
    KB_ARENA_RESULTS_PATH=/home/user/demo/results \
    KB_ARENA_DATASETS_PATH=/home/user/demo/datasets \
    KB_ARENA_CHROMA_PATH=/tmp/chroma

WORKDIR /home/user/demo

EXPOSE 8000

CMD ["uvicorn", "kb_arena.chatbot.api:app", "--host", "0.0.0.0", "--port", "8000"]
