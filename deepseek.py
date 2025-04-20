import os
import csv
import asyncio
from openai import AsyncOpenAI

# Initialize async OpenAI client for DeepSeek
client = AsyncOpenAI(
    api_key="sk-6710f27115c1440fbcfe072c5deeff7a",
    base_url="https://api.deepseek.com"
)

# Optimized prompt template: no unnecessary quotes, include relations
ENTITY_TEMPLATE = (
    "Write a concise paragraph about {entity} in less than 40 words. "
    "Start with a clear definition using the entity's own terms; include these relations: {relations}. "
    "Use appositives or semicolons for structure; avoid lists, markdown, and explicit technical labels. "
    "Example format: [Entity] refers to [definition]; [contextual relationships]."
)

CONCURRENCY_LIMIT = 5  # adjust based on API quotas

async def process_entity(semaphore: asyncio.Semaphore, entity: str, relations: str, writer) -> None:
    """
    Generate description for a single entity and write to CSV via thread to avoid blocking.
    """
    prompt = ENTITY_TEMPLATE.format(entity=entity, relations=relations)

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a knowledge graph expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            desc = response.choices[0].message.content.strip()
        except Exception as e:
            desc = f"Error: {e}"

    # Offload CSV writing to avoid blocking event loop
    await asyncio.to_thread(writer.writerow, {"Entity": entity, "Description": desc})

async def main():
    # Prepare CSV writer
    with open("entity_relations.txt", "r", encoding="utf-8") as infile, \
         open("entity_descriptions.csv", "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["Entity", "Description"])
        writer.writeheader()

        # Semaphore to limit concurrency
        semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
        tasks = []
        block = []

        # Stream and parse blocks to save memory
        for line in infile:
            line = line.strip()
            if line:
                block.append(line)
            else:
                if block:
                    # Handle possible "Center node" prefix
                    if block[0].startswith("Center node:"):
                        center_line = block.pop(0)
                        entity = center_line.split(":", 1)[1].split(",")[0].strip()
                    else:
                        entity = block[0].split(",")[0].strip()
                    relations = " ".join(block)
                    tasks.append(process_entity(semaphore, entity, relations, writer))
                    block = []
        # last block
        if block:
            if block[0].startswith("Center node:"):
                center_line = block.pop(0)
                entity = center_line.split(":", 1)[1].split(",")[0].strip()
            else:
                entity = block[0].split(",")[0].strip()
            relations = " ".join(block)
            tasks.append(process_entity(semaphore, entity, relations, writer))

        # Gather all tasks
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
    print("All entity descriptions have been saved to entity_descriptions.csv")
