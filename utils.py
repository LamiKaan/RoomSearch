import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
import json

def object_to_dict(obj):
    """Recursively converts an object to a dictionary, handling custom objects."""
    if isinstance(obj, dict):
        return {k: object_to_dict(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):  # Handles class instances
        return {k: object_to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [object_to_dict(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(object_to_dict(i) for i in obj)
    elif isinstance(obj, set):
        return {object_to_dict(i) for i in obj}
    else:
        return obj  # Base case, return as is

# Function to print objects with lots of nested attributes in a readable format
def pretty_print_object(obj):
    """Prints an object as a nicely formatted JSON-like structure with top-level variable names."""
   
    # if isinstance(obj, dict) or isinstance(obj, list) or isinstance(obj, tuple) or isinstance(obj, set):
    #     formatted_output = object_to_dict(obj)
    if isinstance(obj, dict) or isinstance(obj, list):
        formatted_output = object_to_dict(obj)
    else:
        # Ensure top-level attributes are explicitly labeled
        formatted_output = {
            attr_name: object_to_dict(getattr(obj, attr_name))
            for attr_name in dir(obj)
            if not attr_name.startswith("__") and not callable(getattr(obj, attr_name))
        }

    print(json.dumps(formatted_output, indent=4, default=str))