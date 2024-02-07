
# Define the context manager for function execution tracking
class FunctionDagContext:
    def __init__(self):
        self.DAG = nx.DiGraph()
        self.current_node = None
        self.latest_id = 0

    def __enter__(self):
        # Nothing to return upon entering the context
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup or resources release if needed when context exits
        pass
    
    def wrap_arg(arg):
        # Generate an identifier for the input
        if isinstance(arg, tuple) and isinstance(arg[0], uuid.UUID):
            input_identifier, *args = args
            self.DAG.add_node(input_identifier)
            # Add input as a node to the DAG
        else:
            input_identifier = uuid.uuid4()
    def dag_wrapped_function(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            input_node = (input_identifier, args, kwargs)
            
            # Execute the Function
            result = func(*args, **kwargs)
            
            # Generate an identifier for the output
            output_identifier = uuid.uuid4()
            # Create output node
            output_node = (output_identifier, result)

            # Add output as a node to DAG
            self.DAG.add_node(output_identifier)
            
            # Add edge representing the function from input node to output node
            self.DAG.add_edge(input_identifier, output_identifier, function=func.__name__)
            
            # Return result with identifier (as tuple to make it a single return value)
            return output_node
        return wrapper    
    
    # def dag_wrapped_function(self, func):

    #     node_map = {}
    #     @wraps(func)
    #     def wrapper(*args, **kwargs):
    #         if len(args)==1:
    #             current_id = self.latest_id
    #             value = args
    #             self.latest_id += 1
    #         else:
    #             current_id, *value = args
    #         # Before Function Execution: Update DAG
    #         previous_node = node_map.get(current_id)
    #         self.current_node = func.__name__
    #         node_map[current_id] = self.current_node
    #         if previous_node is not None:
    #             self.DAG.add_edge(previous_node, self.current_node)
    #         else:
    #             # Add the node if it does not depend on a previous function
    #             self.DAG.add_node(self.current_node)
                
    #         # Execute the Function
    #         result = current_id, func(*value, **kwargs)
            
    #         return result
    #     return wrapper