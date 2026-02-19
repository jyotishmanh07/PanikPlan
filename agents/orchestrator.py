from typing import Annotated, TypedDict, List
import operator
import os
from langgraph.graph import StateGraph, END
import streamlit as st
from tavily import TavilyClient

# Put the key directly in the quotes
tavily = TavilyClient(api_key="blabla")
class PlanState(TypedDict):
    destination: str
    budget_level: int
    candidates: List[dict]
    top_picks: Annotated[list, operator.add]

# Node: Find real restaurants via search
def search_restaurants_node(state: PlanState):
    budget_label = {1: "cheap", 2: "moderately priced", 3: "expensive"}[state['budget_level']]
    query = f"top rated {budget_label} restaurants in {state['destination']} with reviews"

    search_results = tavily.search(query=query, max_results=5, search_depth="advanced") 
    
    formatted_candidates = []
    for res in search_results['results']:
        import random
        fake_reviews = random.randint(50, 500) 
        
        formatted_candidates.append({
            "name": res['title'],
            "price_score": state['budget_level'],
            "number_of_reviews": fake_reviews, 
            "is_unknown": 0, 
            "url": res['url'],
            "content": res['content'] 
        })
    return {"candidates": formatted_candidates}

# Node: Score with trained RandomForest model gotta get the number key from runs each time
def scoring_node(state: PlanState):
    scored_results = []
    for c in state['candidates']:
        c['predicted_rating'] = 4.5 
        scored_results.append(c)
    
    return {"top_picks": scored_results}

# 4. Build the Graph
builder = StateGraph(PlanState)
builder.add_node("search", search_restaurants_node)
builder.add_node("score", scoring_node)
builder.set_entry_point("search")
builder.add_edge("search", "score")
builder.add_edge("score", END)

planner_agent = builder.compile()