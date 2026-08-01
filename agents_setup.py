from typing import Any

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from db import get_supabase_client

load_dotenv()

MODEL = "gemini-3.6-flash"


def _get_opportunities(table_name: str, limit: int) -> list[dict[str, Any]]:
    """Read seeded opportunities without assuming a specific table schema."""
    safe_limit = 10
    response = (
        get_supabase_client()
        .table(table_name)
        .select("*")
        .limit(safe_limit)
        .execute()
    )
    return response.data or []


def get_scholarships(limit: int = 25) -> list[dict[str, Any]]:
    """Return scholarship records from Supabase for matching against a student profile."""
    return _get_opportunities("scholarships", limit)


def get_internships(limit: int = 25) -> list[dict[str, Any]]:
    """Return internship records from Supabase for matching against a student profile."""
    return _get_opportunities("internships", limit)


scholarships_tool = FunctionTool(func=get_scholarships)
internships_tool = FunctionTool(func=get_internships)

parser_agent = LlmAgent(
    name="parser_agent",
    model=MODEL,
    description="Extracts a structured student profile from a request.",
    instruction="""
    Extract the student's education, field of study, graduation year, location,
    skills, interests, financial needs, and career goals from the request.
    State missing information clearly. Return a concise profile that another agent
    can use to select opportunities.
    """,
)

matcher_agent = LlmAgent(
    name="matcher_agent",
    model=MODEL,
    description="Finds scholarships and internships that fit a student profile.",
    instruction="""
    Match the supplied student profile to real records from Supabase. Always call
    both get_scholarships and get_internships before recommending opportunities.
    Recommend only records returned by the tools. Do not invent opportunities,
    deadlines, awards, or URLs.

    Return ONLY valid JSON with no markdown, prose, or code fences, using exactly
    this schema:
    {
      "matches": [
        {
          "title": "string",
          "type": "Scholarship or Internship",
          "why_matched": "string",
          "amount_or_stipend": "string",
          "deadline_or_duration": "string",
          "link": "string"
        }
      ]
    }
    Use "Not specified" for missing record fields. Include eligibility caveats in
    "why_matched". Return an empty matches array if no records fit.
    """,
    tools=[scholarships_tool, internships_tool],
)

roadmap_agent = LlmAgent(
    name="roadmap_agent",
    model=MODEL,
    description="Creates practical next steps from opportunity matches.",
    instruction="""
    Based on the student profile and opportunity matches supplied in the
    conversation, provide a short, practical application roadmap. Prioritize the
    strongest match, include preparation actions, and distinguish confirmed data
    from details the student should verify.

    Return ONLY valid JSON with no markdown, prose, or code fences, using exactly
    this schema:
    {
      "missing_skills": ["string"],
      "roadmap": [
        {
          "step": "string",
          "task": "string",
          "resource": "string"
        }
      ]
    }
    Use an empty missing_skills array when there are no clear gaps.
    """,
)

supervisor_agent = LlmAgent(
    name="supervisor_agent",
    model=MODEL,
    description="Coordinates profile parsing, opportunity matching, and career guidance.",
    instruction="""
    You are OpportunityMatch AI, a career-guidance assistant for Indian students.
    For a student profile or opportunity-matching request, delegate to
    parser_agent and then matcher_agent. Return the matcher_agent's JSON result
    unchanged: no markdown, prose, code fences, or additional keys.

    For a skill-gap or roadmap request, delegate to roadmap_agent. Return the
    roadmap_agent's JSON result unchanged: no markdown, prose, code fences, or
    additional keys. Never fabricate database records or alter agent output.
    """,
    sub_agents=[parser_agent, matcher_agent, roadmap_agent],
)
