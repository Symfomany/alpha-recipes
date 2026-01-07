import chromadb

client = chromadb.PersistentClient(path="chroma_mistral_cgu")
coll = client.get_collection("mistral_cgu")

res = coll.get(
    include=["documents", "metadatas"],  # pas "ids"
    limit=5,
)

print("IDs:", res["ids"])  # les ids sont toujours renvoyés même sans les demander

for i in range(len(res["ids"])):
    print(f"\n=== ID {res['ids'][i]} ===")
    print("META:", res["metadatas"][i])
    print("DOC:", res["documents"][i][:100], "...")
