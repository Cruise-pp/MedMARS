"""
MCP Medical Knowledge Server
=============================
Exposes MedQuAD VectorRAG and DrugBank GraphRAG as standardized MCP tools.

Any MCP-compatible client (Claude Desktop, Claude Code, Cursor, custom agents)
can discover and call these tools without any custom integration.

Usage:
    # stdio mode (for Claude Desktop / Claude Code)
    python mcp_medical_server.py

Architecture:
    ┌─────────────────────────────────────────┐
    │  MCP Medical Server (this file)         │
    │                                         │
    │  Tool: medquad_search                   │
    │    → medquad_rag/query_index.search()   │
    │                                         │
    │  Tool: drugbank_resolve                 │
    │    → drugbank_graph/drugbank_query.*    │
    │                                         │
    │  Tool: drugbank_interaction             │
    │    → drugbank_graph/drugbank_query.*    │
    └────────────────┬────────────────────────┘
                     │ stdio
                     ▼
              MCP Client (Claude, etc.)
"""

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so our modules are importable
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP

# ================================================================
# Initialize MCP server
# ================================================================
mcp = FastMCP(
    "Medical Knowledge Server",
    instructions=(
        "This server provides medical knowledge retrieval tools. "
        "Use medquad_search for general medical Q&A, "
        "drugbank_resolve to look up drug information, "
        "and drugbank_interaction to check drug-drug interactions."
    ),
)


# ================================================================
# Tool 1: MedQuAD VectorRAG Search
# ================================================================
@mcp.tool()
def medquad_search(query: str, top_k: int = 5) -> str:
    """Search the MedQuAD medical knowledge base using hybrid retrieval (FAISS + BM25).

    Use this tool when a user asks general medical questions such as:
    - "What is type 2 diabetes?"
    - "What are the symptoms of pneumonia?"
    - "How is hypertension treated?"

    Args:
        query: The medical question to search for.
        top_k: Number of results to return (default: 5, max: 10).

    Returns:
        JSON string containing matched Q&A pairs with relevance scores.
    """
    from medquad_rag import query_index as mq

    top_k = min(max(top_k, 1), 10)
    results = mq.search(query, top_k=top_k)

    output = {
        "query": query,
        "num_results": len(results),
        "results": [
            {
                "rank": i + 1,
                "question": r["question"],
                "answer": r["answer"],
                "relevance_score": round(r["score"], 4),
            }
            for i, r in enumerate(results)
        ],
    }
    return json.dumps(output, ensure_ascii=False, indent=2)


# ================================================================
# Tool 2: DrugBank Drug Lookup
# ================================================================
@mcp.tool()
def drugbank_resolve(drug_name: str) -> str:
    """Look up a drug by name in the DrugBank database.

    Supports exact match, prefix match (e.g. "Aspiri"), and fuzzy match
    (e.g. typos like "Bihydroergotamine"). Returns drug ID, name,
    indication, and description.

    Use this tool when a user asks about a specific drug, such as:
    - "What is ibuprofen used for?"
    - "Tell me about metformin"

    Args:
        drug_name: The drug name to look up (case-insensitive).

    Returns:
        JSON string with drug information including ID, name, indication,
        and description. Returns candidates if the name is ambiguous.
    """
    from drugbank_graph import drugbank_query as dq

    resolve_result = dq.resolve(drug_name)

    if resolve_result["status"] == "not_found":
        return json.dumps({
            "query": drug_name,
            "status": "not_found",
            "message": f"No drug found matching '{drug_name}'.",
        }, ensure_ascii=False, indent=2)

    # Get full info for the top candidate
    top = resolve_result["candidates"][0]
    drug_info = dq.get_drug(top["drug_id"])
    drug_data = drug_info.get("drug") or {}

    output = {
        "query": drug_name,
        "status": resolve_result["status"],
        "match_mode": resolve_result["mode"],
        "drug": {
            "drug_id": top["drug_id"],
            "name": top["name"],
            "description": drug_data.get("description", ""),
            "indication": drug_data.get("indication", ""),
        },
    }

    # Include other candidates if ambiguous
    if len(resolve_result["candidates"]) > 1:
        output["other_candidates"] = [
            {"drug_id": c["drug_id"], "name": c["name"]}
            for c in resolve_result["candidates"][1:5]
        ]

    return json.dumps(output, ensure_ascii=False, indent=2)


# ================================================================
# Tool 3: DrugBank Drug-Drug Interaction Check
# ================================================================
@mcp.tool()
def drugbank_interaction(drug_a: str, drug_b: str) -> str:
    """Check for known drug-drug interactions (DDI) between two drugs.

    Use this tool when a user asks about combining medications, such as:
    - "Can I take ibuprofen and aspirin together?"
    - "Are there interactions between metformin and lisinopril?"

    Args:
        drug_a: Name of the first drug.
        drug_b: Name of the second drug.

    Returns:
        JSON string with interaction details including evidence descriptions.
        Returns "no_interaction" if no known interactions are found.
    """
    from drugbank_graph import drugbank_query as dq

    # Resolve both drug names
    ra = dq.resolve(drug_a)
    rb = dq.resolve(drug_b)

    # Check if either drug was not found
    not_found = []
    if ra["status"] == "not_found":
        not_found.append(drug_a)
    if rb["status"] == "not_found":
        not_found.append(drug_b)

    if not_found:
        return json.dumps({
            "drug_a": drug_a,
            "drug_b": drug_b,
            "status": "drug_not_found",
            "not_found": not_found,
            "message": f"Could not find: {', '.join(not_found)}",
        }, ensure_ascii=False, indent=2)

    a_id = ra["candidates"][0]["drug_id"]
    a_name = ra["candidates"][0]["name"]
    b_id = rb["candidates"][0]["drug_id"]
    b_name = rb["candidates"][0]["name"]

    # Query interactions
    ddi = dq.ddi_between(a_id, b_id)

    if ddi["status"] == "not_found":
        return json.dumps({
            "drug_a": {"name": a_name, "drug_id": a_id},
            "drug_b": {"name": b_name, "drug_id": b_id},
            "status": "no_interaction",
            "message": f"No known interactions between {a_name} and {b_name}.",
        }, ensure_ascii=False, indent=2)

    return json.dumps({
        "drug_a": {"name": a_name, "drug_id": a_id},
        "drug_b": {"name": b_name, "drug_id": b_id},
        "status": "interaction_found",
        "num_evidence": len(ddi["evidence"]),
        "evidence": ddi["evidence"][:5],  # Limit to top 5 evidence entries
    }, ensure_ascii=False, indent=2)


# ================================================================
# Run the server
# ================================================================
if __name__ == "__main__":
    mcp.run()
