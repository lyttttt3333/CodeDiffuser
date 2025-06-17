import numpy as np
from openai import OpenAI


def from_text_to_embedding(text):
    client = OpenAI(api_key="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")
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    embedding = response.data[0].embedding
    embedding = np.array(embedding)
    return embedding


if __name__ == "__main__":
    text = text = "Hang the red mug on the top branch."
    embedding = from_text_to_embedding(text)
    print(embedding.shape)
