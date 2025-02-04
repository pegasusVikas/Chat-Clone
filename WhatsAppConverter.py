import re
import json
from datetime import datetime
from typing import List, Dict, Tuple

class WhatsAppConverter:
    def __init__(self, friend_name: str):
        """
        Initialize converter with friend's name whose messages we want to capture
        """
        self.friend_name = friend_name
        # Update the timestamp pattern to match "dd/mm/yyyy, hh:mm -"
        self.timestamp_pattern = r'\d{2}/\d{2}/\d{4}, \d{2}:\d{2} -'
        
    def read_chat_file(self, file_path: str) -> List[str]:
        """Read the WhatsApp chat export file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.readlines()
    
    def parse_message(self, line: str) -> Tuple[str, str, str]:
        """Parse a single message line into timestamp, sender, and content"""
        # Extract timestamp
        timestamp_match = re.match(self.timestamp_pattern, line)
        if not timestamp_match:
            return None, None, None
        
        timestamp = timestamp_match.group(0)
        remaining_text = line[len(timestamp):].strip()
        
        # Split into sender and message
        try:
            sender, message = remaining_text.split(':', 1)
            return timestamp, sender.strip(), message.strip()
        except ValueError:
            return None, None, None
    
    def create_conversation_pairs(self, messages: List[str]) -> List[Dict[str, str]]:
        """Create input-output pairs from messages"""
        conversation_pairs = []
        previous_message = None
        previous_sender = None
        
        for line in messages:
            parsed = self.parse_message(line)
            if parsed == (None, None, None):
                # If this is a continuation of the previous message
                if previous_message is not None:
                    if sender == self.friend_name:
                        conversation_pairs[-1]["output"] += " " + line.strip()
                    else:
                        previous_message += " " + line.strip()
                continue
         
            timestamp, sender, message = parsed
            sender = sender.strip()

            # If this message is from our target friend and there was a previous message
            if sender == self.friend_name and previous_message is not None and previous_sender != self.friend_name:
                conversation_pairs.append({
                    "input": previous_message,
                    "output": message
                })
            
            if previous_sender is not None and previous_sender == sender:
                if sender == self.friend_name and len(conversation_pairs) > 0:
                    conversation_pairs[-1]["output"] += ". " + message
                elif sender != self.friend_name:
                    # If the current message is from the same sender as the previous message
                    previous_message += ". " + message
                continue
            
            previous_message = message
            previous_sender = sender
        
        return conversation_pairs
    
    def clean_message(self, message: str) -> str:
        """Clean message content"""
        # Remove Media messages and edited messages tags
        message = re.sub(r'<Media omitted>|<This message was edited>', '', message)
        # Remove URLs (optional)
        message = re.sub(r'http\S+|www.\S+', '', message)
        # Remove multiple spaces
        message = re.sub(r'\s+', ' ', message)
        return message.strip()
    
    def convert_chat(self, input_file: str, output_file: str):
        """Convert WhatsApp chat file to training format"""
        # Read chat file
        messages = self.read_chat_file(input_file)
        
        # Create conversation pairs
        pairs = self.create_conversation_pairs(messages)
        
        # Clean the pairs
        cleaned_pairs = []
        for pair in pairs:
            cleaned_input = self.clean_message(pair["input"])
            cleaned_output = self.clean_message(pair["output"])
            
            # Only keep pairs where both messages are non-empty
            if cleaned_input and cleaned_output:
                cleaned_pairs.append({
                    "input": cleaned_input,
                    "output": cleaned_output
                })
        
        # Save to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"conversations": cleaned_pairs}, f, 
                     ensure_ascii=False, indent=2)

# Example usage
if __name__ == "__main__":
    # Example usage of the converter
    converter = WhatsAppConverter(friend_name="Vikas Gangadevi")  # Replace with your friend's name
    converter.convert_chat(
        input_file="whatsapp_chat.txt",
        output_file="training_data.json"
    )