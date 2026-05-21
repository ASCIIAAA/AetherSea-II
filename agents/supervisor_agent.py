# agents/supervisor_agent.py
import os
from google import genai
from google.genai import types

class SupervisorAgent:
    def __init__(self):
        """
        Initializes the Gemini Client. Ensure you have your API key set in your environment:
        Windows CMD: set GEMINI_API_KEY="your_api_key_here"
        Windows PowerShell: $env:GEMINI_API_KEY="your_api_key_here"
        Linux/Mac: export GEMINI_API_KEY="your_api_key_here"
        """
        # The genai.Client() automatically looks for the GEMINI_API_KEY environment variable.
        self.client = genai.Client()
        # We use gemini-2.5-flash as it is lightning fast and perfect for structured operational text generation
        self.model_name = "gemini-2.5-flash"

    def generate_dispatch_briefing(self, hotspots_count, total_distance, waypoint_list):
        """
        Takes raw outputs from the clustering and routing engines and generates a concise command brief.
        
        hotspots_count: int (Number of verified DBSCAN clusters)
        total_distance: float (Total Haversine route distance in km)
        waypoint_list: list of tuples/lists (The optimized sequence of coordinates)
        """
        
        # 1. Structure the mathematical payload into a clean context block
        data_payload = f"""
        LIVE TELEMETRY SUMMARY:
        - Verified Floating Debris Clusters Found: {hotspots_count}
        - Total Optimized Mission Distance: {total_distance:.2f} kilometers
        - Scheduled Interception Waypoints (In Order): {waypoint_list}
        """

        # 2. Design the system instructions to force Gemini to act like an operational dispatcher
        system_instruction = """
        You are the Senior Automated Maritime Dispatcher for AetherSea, an AI-powered marine cleanup system. 
        Your job is to translate raw coordinate arrays and telemetry data into professional, high-urgency, 
        and actionable naval deployment briefs. 
        
        CRITICAL RULES:
        1. Do NOT hallucinate coordinates or data. Use ONLY the exact numbers provided.
        2. Keep the briefing highly concise and professional—harbor crew and captains read this.
        3. Do not use conversational filler (e.g., avoid "Sure, here is the brief"). Jump straight into the report.
        """

        # 3. Create the user prompt
        user_prompt = f"""
        Review the following telemetry data and generate an operational mission brief:
        {data_payload}
        
        Please format your output strictly with these three sections:
        🚨 SITUATION ASSESSMENT (Highlight the severity based on the cluster count)
        🚢 NAVIGATION & LOGISTICS (Summarize the route distance and travel order)
        📋 DIRECT ACTION REQUIRED (Provide a clear next-step instruction for the harbor crew)
        """

        try:
            # 4. Execute the API call
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2, # Low temperature ensures strict factual adherence to your math
                )
            )
            return response.text
            
        except Exception as e:
            return f"LOGISTICS ENGINE ERROR: Failed to synthesize dispatch brief. Details: {str(e)}"