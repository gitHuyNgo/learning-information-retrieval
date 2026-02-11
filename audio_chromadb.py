import json
import chromadb
from faster_whisper import WhisperModel
from openai import OpenAI

class AudioSegmenter:
  def __init__(self, openai_api_key, db_path="./chromadb", collection_name="audio"):
    self.client = OpenAI(api_key=openai_api_key)
    self.whisper = WhisperModel("large-v3", device="cuda", compute_type="float16")
    self.embedding_model = "text-embedding-3-large"

    self.chromadb_client = chromadb.PersistentClient(path=db_path)
    self.collection = self.chromadb_client.get_or_create_collection(name=collection_name)

  def process_video(self, video_id, audio_path, max_duration=60, batch_size=40):
    raw_data = self._transcribe(audio_path)
    if not raw_data:
      return

    break_points = self._get_granular_breaks(raw_data, batch_size)
    chunks = self._finalize_chunks(raw_data, break_points, max_duration)

    if not chunks:
      return

    chunks = self._enrich_metadata_batched(chunks)

    ids = [f"{video_id}_{i}" for i in range(len(chunks))]

    embeddings = [c["embedding"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{
      "video_id": video_id,
      "start": c["start"],
      "end": c["end"],
      "keywords": c.get("keywords", "")
    } for c in chunks]

    self.collection.upsert(
      ids=ids,
      embeddings=embeddings,
      metadatas=metadatas,
      documents=documents
    )

  def search(self, query, video_id=None, top_k=3):
    q_emb = self.client.embeddings.create(
      input=[query],
      model=self.embedding_model
    ).data[0].embedding

    where = {"video_id": video_id} if video_id else None

    results = self.collection.query(
      query_embeddings=[q_emb],
      n_results=top_k,
      where=where
    )

    formatted_results = []
    ids = results.get("ids", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    documents = results.get("documents", [[]])[0] if "documents" in results else [None] * len(ids)

    for i in range(len(ids)):
      formatted_results.append({
        "id": ids[i],
        "distance": distances[i],
        "metadata": metadatas[i],
        "document": documents[i]
    })

    return formatted_results

  def _transcribe(self, audio_path):
    segments, _ = self.whisper.transcribe(
        audio_path,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=1000)
    )

    return [{
        "id": i,
        "text": seg.text.strip(),
        "start": round(float(seg.start), 2),
        "end": round(float(seg.end), 2)
    } for i, seg in enumerate(segments)]

  def _get_granular_breaks(self, full_data, batch_size):
    if not full_data:
        return []

    all_breaks = []
    for i in range(0, len(full_data), batch_size):
        batch = full_data[i: i + batch_size]
        if not batch:
            continue

        numbered_text = "\n".join([f"[{s['id']}] {s['text']}" for s in batch])

        prompt = (
            "Split this transcript into chunks targeting ~30–60 seconds each. "
            "Return JSON with key end_ids: a list of segment ids that should be the END of a chunk. "
            "Rules: ids must be from the provided list; must be increasing; include the last id of this batch.\n\n"
            f"{numbered_text}"
        )

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        payload = json.loads(response.choices[0].message.content)
        end_ids = payload.get("end_ids", [])

        for eid in end_ids:
            if isinstance(eid, int):
                all_breaks.append(eid)

        all_breaks.append(batch[-1]["id"])

    return sorted(set(all_breaks))

  
  def _finalize_chunks(self, data, breaks, max_dur):
    if not data:
        return []

    chunks = []
    start_idx = 0

    breaks = sorted(set(list(breaks) + [len(data) - 1]))

    i = 0
    while i < len(breaks):
        b_id = breaks[i]

        if b_id < start_idx:
            i += 1
            continue

        slice_ = data[start_idx: b_id + 1]
        if not slice_:
            i += 1
            continue

        dur = slice_[-1]["end"] - slice_[0]["start"]

        if dur > max_dur and len(slice_) > 1:
            t_end = start_idx
            base_start = slice_[0]["start"]

            for s in slice_:
                if (s["end"] - base_start) > max_dur:
                    break
                t_end = s["id"]

            if t_end < start_idx:
                t_end = start_idx

            b_id = t_end
            slice_ = data[start_idx: b_id + 1]

            if b_id not in breaks:
                breaks.insert(i, b_id)

        chunks.append({
            "text": " ".join(s["text"] for s in slice_).strip(),
            "start": slice_[0]["start"],
            "end": slice_[-1]["end"],
        })

        start_idx = b_id + 1
        i += 1

        if start_idx >= len(data):
            break

    return chunks


  def _enrich_metadata_batched(self, chunks, embed_batch_size=20):
    if not chunks:
      return []

    texts = [c["text"] for c in chunks]

    for i in range(0, len(texts), embed_batch_size):
      batch_texts = texts[i: i + embed_batch_size]
      response = self.client.embeddings.create(
        input=batch_texts,
        model=self.embedding_model
      )
      for j, item in enumerate(response.data):
        chunks[i + j]["embedding"] = item.embedding

    for chunk in chunks:
      kw_res = self.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
          "role": "user",
          "content": f"Extract 3 technical keywords: {chunk['text']}"
        }]
    )
      chunk["keywords"] = kw_res.choices[0].message.content.strip()

    return chunks