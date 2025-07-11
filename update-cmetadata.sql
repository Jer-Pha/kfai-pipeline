UPDATE langchain_pg_embedding AS emb
SET cmetadata = emb.cmetadata || jsonb_build_object(
    'show_name', v.show_name,
    'hosts', v.hosts,
    'published_at', v.published_at
)
FROM video_transcript_chunks AS c
JOIN videos AS v ON v.video_id = c.video_id
WHERE emb.document = c.chunk_text
  AND (
    NOT emb.cmetadata ? 'show_name'
    OR NOT emb.cmetadata ? 'hosts'
    OR NOT emb.cmetadata ? 'published_at'
  );