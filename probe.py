from fastembed import TextEmbedding

print("import ok", flush=True)
m = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("model loaded", flush=True)
v = next(m.embed(["hello world"]))
print("embed ok, dim =", len(v.tolist()), flush=True)
