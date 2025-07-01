import os
import json
import re
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from flightmodel import predict_price_from_input


load_dotenv()
llm = ChatOllama(model="llama3", temperature=0)

# --- Agent State ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], "Chat messages history"]
    parsed_input: Annotated[dict, "Parsed flight details"]
    missing_fields: Annotated[list[str], "List of fields missing from parsed_input"]
    all_prices: Annotated[list[dict], "List of predicted prices for different airlines"]

# --- Constants ---
REQUIRED_FIELDS = ['source_city', 'departure_time', 'stops',
                   'arrival_time', 'destination_city', 'class', 'days_left']

# Default duration if not provided by the user
DEFAULT_DURATION_HOURS = 2
AIRLINE_NAME_MAP = {
    "indigo": "Indigo",
    "air india": "Air_India",
    "air_india": "Air_India",
    "spicejet": "SpiceJet",
    "airasia": "AirAsia",
    "air asia": "AirAsia",
    "vistara": "Vistara",
    "go first": "GO_FIRST",
    "go_first": "GO_FIRST",
    "gofirst": "GO_FIRST", # Common variation
    "goair": "GO_FIRST",   # Another common variation
}
VALID_AIRLINES = list(set(AIRLINE_NAME_MAP.values()))


# --- Nodes ---

# Step 1: Extract parameters from user text
def extract_parameters(state: AgentState) -> AgentState:
    messages = state["messages"]

    # Updated prompt to explicitly NOT extract 'airline'
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that extracts flight details from a conversation.
Your task is to consolidate ALL flight details provided by the user across the entire conversation history.
Extract these flight details and return ONLY valid JSON with exact fields.
Use double quotes for all strings, no comments, no trailing commas, and no extra text outside the JSON object.
Fields: source_city, departure_time, stops, arrival_time, destination_city, class, duration, days_left.
Do NOT try to extract 'airline' as it will be iterated over for comparison.

If a field has been mentioned at any point in the conversation, include its value.
If a field is not mentioned at all in the entire conversation, OMIT that field entirely from the JSON object.
Do NOT include fields with empty strings, null, or placeholder values if a valid value was not provided.
If new information is provided for a field, overwrite the old value.

Example output for a conversation where user said "I want a flight from Delhi" then "to Mumbai":
{{
  "source_city": "Delhi",
  "destination_city": "Mumbai"
}}

Respond with valid JSON only, nothing else. If no relevant information is found, return an empty JSON object {{}}."""),
        *messages
    ])

    chain = prompt | llm
    output = chain.invoke({})
    raw_response = output.content
    print("🧠 LLM raw response (extract_parameters):", raw_response)

    extracted_data = {}
    try:
        extracted_data = json.loads(raw_response)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', raw_response, flags=re.DOTALL)
        if json_match:
            candidate_json = json_match.group()
            candidate_json = re.sub(r",(\s*[}\]])", r"\1", candidate_json)
            candidate_json = candidate_json.replace("'", '"')
            try:
                extracted_data = json.loads(candidate_json)
            except Exception as e:
                print(f"Error during second JSON parse attempt: {e}")
                extracted_data = {}
        else:
            print("Could not find JSON object in LLM output.")
            extracted_data = {}

    filtered_extracted_data = {
        k: v for k, v in extracted_data.items()
        if v is not None and str(v).strip() != ""
    }

    current_parsed_input = state.get("parsed_input", {})
    updated_parsed_input = {**current_parsed_input, **filtered_extracted_data}

    # Only include fields relevant to our process (REQUIRED_FIELDS + 'duration' if extracted)
    # Remove 'airline' if it was accidentally extracted, as we'll iterate through all of them later.
    all_relevant_fields = set(REQUIRED_FIELDS + ['duration'])
    final_parsed_input = {k: v for k, v in updated_parsed_input.items() if k in all_relevant_fields}
    
    # Ensure 'airline' is not in the final parsed input as it will be set by the system
    if 'airline' in final_parsed_input:
        del final_parsed_input['airline']


    return {
        "messages": messages,
        "parsed_input": final_parsed_input,
        "all_prices": [], # Initialize as empty list for new prediction cycle
        "missing_fields": []
    }

# Step 2: Check for missing parameters
def check_missing_parameters(state: AgentState) -> AgentState:
    parsed_input = state.get("parsed_input", {})
    # Only check for REQUIRED_FIELDS that the user must provide
    missing = [key for key in REQUIRED_FIELDS if key not in parsed_input or parsed_input[key] is None or parsed_input[key] == ""]

    return {
        "messages": state["messages"],
        "parsed_input": parsed_input,
        "missing_fields": missing,
        "all_prices": [] # Reset or keep existing, depends on flow. Resetting for new cycle.
    }

# Step 3: Ask for missing fields
def ask_for_missing_fields(state: AgentState) -> AgentState:
    messages = state["messages"]
    missing_fields = state["missing_fields"]

    if not missing_fields:
        return {
            "messages": messages + [AIMessage(content="All necessary information is present. Proceeding with price comparison.")],
            "parsed_input": state["parsed_input"],
            "missing_fields": [],
            "all_prices": state["all_prices"]
        }

    field_names = {
        # 'airline' is explicitly removed from here
        'source_city': 'departure city',
        'departure_time': 'departure time (e.g., Morning, Evening, Night)',
        'stops': 'number of stops (e.g., zero, one, two+)',
        'arrival_time': 'arrival time (e.g., Morning, Afternoon, Night)',
        'destination_city': 'destination city',
        'class': 'class (e.g., Economy, Business)',
        'days_left': 'how many days from now you plan to travel'
        # 'duration' is intentionally excluded here
    }

    questions = []
    for field in missing_fields:
        if field in field_names: # Ensure we only ask for fields that are in our field_names map
            questions.append(f"the {field_names.get(field, field)}")

    if not questions:
        ai_response = "I'm missing some information, but I can't identify what. Can you provide more details?"
    elif len(questions) == 1:
        question_text = f"Could you please tell me {questions[0]}?"
    elif len(questions) == 2:
        question_text = f"Could you please tell me {questions[0]} and {questions[1]}?"
    else:
        last_q = questions.pop()
        question_text = f"I also need {', '.join(questions)}, and {last_q}."

    ai_response = f"I'm missing some information to predict flight prices. {question_text}"

    return {
        "messages": messages + [AIMessage(content=ai_response)],
        "parsed_input": state["parsed_input"],
        "missing_fields": missing_fields,
        "all_prices": state["all_prices"]
    }

# Step 4: Validate and convert input
def validate_input(state: AgentState) -> AgentState:
    parsed_input = state.get("parsed_input", {})
    messages = state["messages"]

    # Before validation, ensure all *REQUIRED_FIELDS* are actually present,
    # and handle the optional 'duration' field.
    for key in REQUIRED_FIELDS:
        if key not in parsed_input or parsed_input[key] is None or str(parsed_input[key]).strip() == "":
            return {
                "messages": messages + [AIMessage(content=f"Internal error: Missing required field '{key}' during validation. This should not happen if previous checks are correct.")],
                "parsed_input": parsed_input,
                "all_prices": state["all_prices"],
                "missing_fields": []
            }

    try:
        # --- Handle 'duration' (OPTIONAL FIELD) ---
        if 'duration' in parsed_input and isinstance(parsed_input['duration'], str):
            duration_match = re.search(r'\d+', parsed_input['duration'])
            if duration_match:
                parsed_input['duration'] = int(duration_match.group())
            else:
                print(f"Warning: Could not extract a valid number for 'duration' from '{parsed_input['duration']}'. Using default.")
                parsed_input['duration'] = DEFAULT_DURATION_HOURS
        elif 'duration' in parsed_input: # If it's already a number, ensure it's int
            parsed_input['duration'] = int(parsed_input['duration'])
        else:
            # If 'duration' was not extracted by LLM at all, apply default
            parsed_input['duration'] = DEFAULT_DURATION_HOURS
            print(f"Info: 'duration' not provided by user/LLM. Defaulting to {DEFAULT_DURATION_HOURS} hours.")

        # --- Handle 'days_left' ---
        parsed_input['days_left'] = int(parsed_input['days_left'])

        # --- Handle 'stops' ---
        if 'stops' in parsed_input:
            if isinstance(parsed_input['stops'], str):
                stops_str = parsed_input['stops'].strip().lower()
                if stops_str == "0" or stops_str == "zero":
                    parsed_input['stops'] = "zero"
                elif stops_str == "1" or stops_str == "one":
                    parsed_input['stops'] = "one"
                elif stops_str in ["2", "two", "two+", "2+"]:
                    parsed_input['stops'] = "two+"
                else:
                    raise ValueError(f"Invalid string value for 'stops': {parsed_input['stops']}. Expected 'zero', 'one', or 'two+'.")
            elif isinstance(parsed_input['stops'], (int, float)):
                if parsed_input['stops'] == 0:
                    parsed_input['stops'] = "zero"
                elif parsed_input['stops'] == 1:
                    parsed_input['stops'] = "one"
                elif parsed_input['stops'] >= 2:
                    parsed_input['stops'] = "two+"
                else:
                    raise ValueError(f"Invalid numeric value for 'stops': {parsed_input['stops']}. Expected 0, 1, or 2+.")

        # --- Normalize other categorical fields ---
        if 'class' in parsed_input and isinstance(parsed_input['class'], str):
            parsed_input['class'] = parsed_input['class'].strip().title()

        for field in ['source_city', 'destination_city', 'departure_time', 'arrival_time']:
            if field in parsed_input and isinstance(parsed_input[field], str):
                parsed_input[field] = parsed_input[field].strip().title()

    except ValueError as ve:
        return {
            "messages": messages + [AIMessage(content=f"Error with input formats: {ve}. Please check your flight details carefully.")],
            "parsed_input": parsed_input,
            "all_prices": state["all_prices"],
            "missing_fields": []
        }
    except Exception as e:
        return {
            "messages": messages + [AIMessage(content=f"An unexpected error occurred during input validation: {e}")],
            "parsed_input": parsed_input,
            "all_prices": state["all_prices"],
            "missing_fields": []
        }

    return {
        "messages": messages,
        "parsed_input": parsed_input,
        "all_prices": state["all_prices"]
    }

# Step 5: Predict prices for all airlines
def predict_all_airlines_prices(state: AgentState) -> AgentState:
    parsed_input = state.get("parsed_input", {})
    messages = state["messages"]
    
    all_predicted_prices = []

    # Ensure all base required fields are present before attempting predictions
    base_fields_for_prediction = REQUIRED_FIELDS + ['duration'] # duration is now guaranteed by validate_input
    for key in base_fields_for_prediction:
        if key not in parsed_input or parsed_input[key] is None or parsed_input[key] == "":
            error_message = f"Internal error: Missing crucial base field '{key}' for prediction. Cannot proceed with airline comparisons."
            return {
                "messages": messages + [AIMessage(content=error_message)],
                "parsed_input": parsed_input,
                "all_prices": [],
                "missing_fields": []
            }

    print("\n--- Calculating prices for all airlines ---")
    for airline_name in VALID_AIRLINES:
        temp_input = parsed_input.copy()
        temp_input['airline'] = airline_name # Add the current airline to the input
        
        try:
            # Call your actual prediction model
            price = predict_price_from_input(temp_input)
            all_predicted_prices.append({'airline': airline_name, 'price': price})
            print(f"  - {airline_name}: ₹{price:,.2f}")
        except Exception as e:
            print(f"  - Warning: Could not predict price for {airline_name} due to error: {e}")
            # Optionally, you could still add it with a 'N/A' price or just skip it.
            # all_predicted_prices.append({'airline': airline_name, 'price': 'N/A'})

    print("--- Finished calculating prices ---\n")

    return {
        "messages": messages,
        "parsed_input": parsed_input,
        "all_prices": all_predicted_prices,
        "missing_fields": []
    }

# Step 6: Rank and present the top fares
def rank_and_present_fares(state: AgentState) -> AgentState:
    messages = state["messages"]
    all_prices = state.get("all_prices", [])

    if not all_prices:
        ai_response = "I couldn't predict prices for any airlines with the given details."
    else:
        # Filter out any 'N/A' prices if you added them, and ensure it's comparable
        comparable_prices = [p for p in all_prices if isinstance(p.get('price'), (int, float))]
        
        if not comparable_prices:
            ai_response = "I couldn't predict valid numerical prices for any airlines with the given details."
        else:
            # Sort by price in ascending order
            sorted_prices = sorted(comparable_prices, key=lambda x: x['price'])
            
            # Select the top 3
            top_3_fares = sorted_prices[:3]

            response_lines = ["Here are the top 3 lowest estimated flight prices based on your requirements:"]
            for i, fare in enumerate(top_3_fares):
                response_lines.append(f"{i+1}. {fare['airline']}: ₹{fare['price']:,.2f}")
            
            ai_response = "\n".join(response_lines)

    return {
        "messages": messages + [AIMessage(content=ai_response)],
        "parsed_input": state["parsed_input"],
        "all_prices": all_prices, # Keep the full list in state for debugging if needed
        "missing_fields": []
    }

# --- Edges and Graph Compilation ---

workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("extract_parameters", extract_parameters)
workflow.add_node("check_missing_parameters", check_missing_parameters)
workflow.add_node("ask_for_missing_fields", ask_for_missing_fields)
workflow.add_node("validate_input", validate_input)
workflow.add_node("predict_all_airlines_prices", predict_all_airlines_prices) # New node
workflow.add_node("rank_and_present_fares", rank_and_present_fares)       # New node

# Set entry point
workflow.set_entry_point("extract_parameters")

# Define conditional edge from check_missing_parameters
def should_ask_for_fields(state: AgentState) -> str:
    if state["missing_fields"]:
        print(f"Condition: Missing fields: {state['missing_fields']}. Asking user.")
        return "ask_for_fields"
    else:
        print("Condition: All fields present. Validating input.")
        return "validate_and_predict"

# Add edges
workflow.add_edge("extract_parameters", "check_missing_parameters")

# Conditional routing from check_missing_parameters
workflow.add_conditional_edges(
    "check_missing_parameters",
    should_ask_for_fields,
    {
        "ask_for_fields": "ask_for_missing_fields",
        "validate_and_predict": "validate_input" # If no missing fields, proceed to validation
    }
)

workflow.add_edge("ask_for_missing_fields", END) # End the current turn for user input

# After validation, go to parallel prediction, then rank and present
workflow.add_edge("validate_input", "predict_all_airlines_prices")
workflow.add_edge("predict_all_airlines_prices", "rank_and_present_fares")
workflow.add_edge("rank_and_present_fares", END) # End the graph after presenting results

app = workflow.compile()
