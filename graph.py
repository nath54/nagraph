#
from typing import Optional, Any, Callable
#
import random


#
### Class that represents a Graph node ###
#
class GraphNode:
    #
    ### Constructor ###
    #
    def __init__(self, node_id: str, graph: Graph, values: Optional[dict[str, Any]] = None) -> None:
        #
        ### Unique ID of the node in the graph ###
        #
        self.node_id: str = node_id
        #
        ### Reference to the node's graph ###
        #
        self.graph: Graph = Graph
        #
        ### Container for values to register in this node ###
        #
        self.values: Optional[dict[str, Any]] = values


#
### Class that represents a graph ###
#
class Graph:
    #
    ### Constructor ###
    #
    def __init__(self, weighted: bool = False) -> None:
        #
        ### Container to quickly store & access all the nodes ###
        #
        self.nodes: dict[str, GraphNode] = {}
        #
        ### Indicates if there are weights stored to the edges ###
        #
        self.weighted: bool = weighted
        #
        ### Container to quickly store & access all the edges with weights ###
        #
        self.nodes_edges: dict[str, set[str] | dict[str, float]] = {}
        #
        ### To access in O(1) to all the edges from destination ###
        #
        self.nodes_edges_sym: dict[str, set[str]] = {}

    #
    ### Function to generate an unique node_id ###
    #
    def generate_unique_node_id(self) -> str:
        #
        ### Constant for id generation, best if larger value to reduce collision and maximum nodes capacity ###
        #
        MAXINT: int = 9999999999
        #
        ### Generates random id ###
        #
        new_id: str = str(random.randint(0, MAXINT))
        #
        ### Guardrail to avoid while loop, maximum iteration limit, and raise error if limit reached ###
        #
        max_iter: int = 1000
        crt_iter: int = 0
        #
        ### While the generated node id already exists, generates a new one, and if max_iter reached, raise an error ###
        #
        while new_id in self.nodes:
            #
            ### Check for max iter limit reached ###
            #
            if crt_iter >= max_iter:
                #
                raise UserWarning("Error: Reached maximum iteration for unique new node id generation !")
            #
            ### Generates a new random node id ###
            #
            new_id = str(random.randint(0, MAXINT))
            #
            ### Increment iteration counter to avoid while loop ###
            #
            crt_iter += 1

    #
    ### Function to add a new node in the graph ###
    #
    def add_node(self, node_id: str = "", values: Optional[dict[str, Any]] = None) -> GraphNode:
        #
        ### If no node_id given, generate an unique one ###
        #
        if node_id == "":
            #
            node_id = self.generate_unique_node_id()
        #
        ### Check for unicity of node_id ###
        #
        if node_id in self.nodes:
            #
            raise UserWarning(f"Error: Trying to add a node but a node with same node_id (=`{node_id}`) already exists in the graph !")
        #
        ### If no errors, add the node to the graph ###
        #
        self.nodes[node_id] = GraphNode(node_id=node_id, graph=self, values=values)
        #
        ### Returns the newly created node ###
        #
        return self.nodes[node_id]

    #
    ### Function to add an edge to the graph ###
    #
    def add_edge(self, src_node_id: str, dst_node_id: str, weight: float = 1, add_symetric: bool = False) -> None:
        #
        ### Check for source node existence ###
        #
        if src_node_id not in self.nodes:
            #
            raise UserWarning(f"Error: Trying to add an edge between two nodes, but node with node_id=`{src_node_id}` does not exist in the graph !")
        #
        ### Check for destination node existence ###
        #
        if dst_node_id not in self.nodes:
            #
            raise UserWarning(f"Error: Trying to add an edge between two nodes, but node with node_id=`{dst_node_id}` does not exist in the graph !")
        #
        ### If first edge from source node, initialize it ###
        #
        if src_node_id not in self.nodes_edges:
            #
            self.nodes_edges[src_node_id] = dict() if self.weighted else set()
        #
        if dst_node_id not in self.nodes_edges_sym:
            #
            self.nodes_edges_sym[dst_node_id] = set()
        #
        ### Add the edge to the graph ###
        #
        if self.weighted:
            #
            self.nodes_edges[src_node_id][dst_node_id] = weight
        #
        else:
            #
            self.nodes_edges[src_node_id].add( dst_node_id )
        #
        self.nodes_edges_sym[dst_node_id].add( src_node_id )
        #
        ### Adds the symetric edge if needed ###
        #
        if add_symetric:
            #
            self.add_edge(src_node_id=dst_node_id, dst_node_id=src_node_id, add_symetric=False)

    #
    ### Function to remove an edge from the graph ###
    #
    def remove_edge(self, src_node_id: str, dst_node_id: str, remove_symetric: bool = False) -> None:
        #
        ### Check for source node existence ###
        #
        if src_node_id not in self.nodes:
            #
            raise UserWarning(f"Error: Trying to remove an edge between two nodes, but node with node_id=`{src_node_id}` does not exist in the graph !")
        #
        ### Check for destination node existence ###
        #
        if dst_node_id not in self.nodes:
            #
            raise UserWarning(f"Error: Trying to remove an edge between two nodes, but node with node_id=`{dst_node_id}` does not exist in the graph !")
        #
        ### Check for edge existence ###
        #
        if (src_node_id not in self.nodes_edges) or (dst_node_id not in self.nodes_edges[src_node_id]):
            #
            raise UserWarning(f"Error: Tring to remove an edge that doesn't exists from node_id={src_node_id} to node_id={dst_node_id}")
        #
        ### Removes the edge ###
        #
        if self.weighted:
            #
            self.nodes_edges[src_node_id].pop( dst_node_id )
        #
        else:
            #
            self.nodes_edges[src_node_id].remove( dst_node_id )
        #
        self.nodes_edges_sym[dst_node_id].remove( src_node_id )
        #
        ### Cleaning edge structure if empty ###
        #
        if len(self.nodes_edges[src_node_id]) == 0:
            #
            del self.nodes_edges[src_node_id]
        #
        if len(self.nodes_edges[dst_node_id]) == 0:
            del self.nodes_edges_sym[dst_node_id]

    #
    ### Function to remove all the edges that have a specific source node from the graph ###
    #
    def remove_all_edges_from_src_node(self, src_node_id: str) -> None:
        #
        ### Check for source node existence ###
        #
        if src_node_id not in self.nodes:
            #
            raise UserWarning(f"Error: Trying to remove an edge between two nodes, but node with node_id=`{src_node_id}` does not exist in the graph !")
        #
        ### If no edges, ends directly the function here ###
        #
        if src_node_id not in self.nodes_edges:
            #
            return
        #
        ### Remove all the edges from the source node ###
        #
        destination_nodes: list[str] = list( self.nodes_edges[src_node_id] )
        #
        for dst_node_id in destination_nodes:
            #
            self.remove_edge(src_node_id=src_node_id, dst_node_id=dst_node_id)

    #
    ### Function to remove all the edges that have a specific destination node from the graph ###
    #
    def remove_all_edges_to_dst_node(self, dst_node_id: str) -> None:
        #
        ### Check for source node existence ###
        #
        if dst_node_id not in self.nodes:
            #
            raise UserWarning(f"Error: Trying to remove an edge between two nodes, but node with node_id=`{dst_node_id}` does not exist in the graph !")
        #
        ### If no edges, ends directly the function here ###
        #
        if dst_node_id not in self.nodes_edges_sym:
            #
            return
        #
        ### Remove all the edges that have the specific destination node ###
        #
        source_nodes: list[str] = list( self.nodes_edges_sym[dst_node_id] )
        #
        for src_node_id in source_nodes:
            #
            self.remove_edge(src_node_id=src_node_id, dst_node_id=dst_node_id)

    #
    ### Function to remove a node from the graph ###
    #
    def remove_node(self, node_id: str, remove_edges_with: bool = True) -> None:
        #
        ### Check for node existence ###
        #
        if node_id not in self.nodes:
            #
            raise UserWarning(f"Error: trying to delete inexisting node in graph !")
        #
        ### Check for edges ###
        #
        has_edges: bool = (node_id in self.nodes_edges and len(self.nodes_edges[node_id]) > 0) or (node_id in self.nodes_edges_sym and len(self.nodes_edges_sym[node_id]) > 0)
        #
        if has_edges:
            #
            ### If removes edges with param ###
            #
            if not remove_edges_with:
                #
                raise UserWarning(f"Error: cannot remove a node from the graph without removing all its edges !")
            #
            ### Remove all the edges with node_id as source ###
            #
            self.remove_all_edges_from_src_node(src_node_id=node_id)
            #
            ### Remove all the edges with node_id as destination ###
            #
            self.remove_all_edges_to_dst_node(dst_node_id=node_id)
        #
        ### Remove the node from the node list ###
        #
        del self.nodes[node_id]

    #
    ### Function to get node ###
    #
    def get_node(self, node_id: str) -> GraphNode:
        #
        ### Check for node existence ###
        #
        if node_id not in self.nodes:
            #
            raise UserWarning(f"Error: node not found with node_id=`{node_id}` in the graph !")
        #
        ### Returns the node ###
        #
        return self.nodes[node_id]

    #
    ### Function to get a specific node value ###
    #
    def get_node_value(self, node_id: str, value_key: str) -> GraphNode:
        #
        ### Check for node existence ###
        #
        if node_id not in self.nodes:
            #
            raise UserWarning(f"Error: node not found with node_id=`{node_id}` in the graph !")
        #
        ### Check for value existence ###
        #
        if value_key not in self.nodes[node_id].values:
            #
            raise UserWarning(f"Error: value not found in node with node_id=`{node_id}` and value_key=`{value_key}` in the graph !")
        #
        ### Returns the value ###
        #
        return self.nodes[node_id].values[value_key]

    #
    ### Function to set a specific node value ###
    #
    def set_node_value(self, node_id: str, value_key: str, value: Any) -> None:
        #
        ### Check for node existence ###
        #
        if node_id not in self.nodes:
            #
            raise UserWarning(f"Error: node not found with node_id=`{node_id}` in the graph !")
        #
        ### Set the node value ###
        #
        self.nodes[node_id].values[value_key] = value

    #
    ### Function to delete a specific node value ###
    #
    def del_node_value(self, node_id: str, value_key: str) -> GraphNode:
        #
        ### Check for node existence ###
        #
        if node_id not in self.nodes:
            #
            raise UserWarning(f"Error: node not found with node_id=`{node_id}` in the graph !")
        #
        ### Check for value existence ###
        #
        if value_key not in self.nodes[node_id].values:
            #
            raise UserWarning(f"Error: value not found in node with node_id=`{node_id}` and value_key=`{value_key}` in the graph !")
        #
        ### Delete the value ###
        #
        del self.nodes[node_id].values[value_key]

    #
    ### Get all the in-neighbors ids ###
    #
    def get_predecessors_ids_of_node(self, node_id: str = "", node: Optional[GraphNode] = None) -> list[str]:
        #
        ### Check for given arguments ###
        #
        if node_id == "" and node is None:
            #
            raise UserWarning(f"Error: Called function `get_predecessors_of_node` without valid arguments !")
        #
        if node is not None:
            #
            node_id = node.node_id
        #
        ### Check for for node existence ###
        #
        if node_id not in self.nodes:
            #
            raise UserWarning(f"Error: The node with node_id=`{node_id}` does't exist in the graph !")
        #
        ### Get all predecessors and returns them ###
        #
        if node_id not in self.nodes_edges_sym:
            #
            return []
        #
        return self.nodes_edges_sym[node_id]

    #
    ### Get all the out-neighbors ids ###
    #
    def get_successors_ids_of_node(self, node_id: str = "", node: Optional[GraphNode] = None) -> list[str]:
        #
        ### Check for given arguments ###
        #
        if node_id == "" and node is None:
            #
            raise UserWarning(f"Error: Called function `get_successors_of_node` without valid arguments !")
        #
        if node is not None:
            #
            node_id = node.node_id
        #
        ### Check for for node existence ###
        #
        if node_id not in self.nodes:
            #
            raise UserWarning(f"Error: The node with node_id=`{node_id}` does't exist in the graph !")
        #
        ### Get all successors and returns them ###
        #
        if node_id not in self.nodes_edges:
            #
            return []
        #
        return self.nodes_edges[node_id]

    #
    ### Get all the in-neighbors ###
    #
    def get_predecessors_of_node(self, node_id: str = "", node: Optional[GraphNode] = None) -> list[GraphNode]:
        #
        return [self.nodes[src_node_id] for src_node_id in self.get_predecessors_ids_of_node(node_id=node_id, node=node)]

    #
    ### Get all the out-neighbors ###
    #
    def get_successors_of_node(self, node_id: str = "", node: Optional[GraphNode] = None) -> list[GraphNode]:
        #
        return [self.nodes[src_node_id] for dst_node_id in self.get_successors_ids_of_node(node_id=node_id, node=node)]

    #
    ### Function that performs a simple graph exploration
    #
    def explore_from_source(
        self,
        src_node_id: str,
        exploration_algorithm: str = "bfs",
        force_sym_edges_to_exists: bool = False,
        nodes_marks: Optional[dict[str, Any]] = None,
        custom_node_ordering: Optional[Callable[str, float]] = None,
        fn_to_mark_nodes: Callable[..., Any] = lambda _: 1,
        fn_to_mark_nodes_args: Optional[list[Any]] = None,
        fn_to_mark_nodes_kwargs: Optional[dict[str, Any]] = None,
        fn_on_loop: Callable[..., bool] = lambda _: True,
        fn_on_loop_args: Optional[list[Any]] = None,
        fn_on_loop_kwargs: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        #
        ### Check for valid algorithm ###
        #
        if not exploration_algorithm in ["bfs", "dfs", "random"]:
            #
            raise UserWarning(f"Error: unknown exploration algorithm = `{exploration_algorithm}`")
        #
        ### Initialize marked nodes, uses given one if given one ###
        #
        nodes_marks: dict[str, Any] = nodes_marks if nodes_marks is not None else {}
        #
        ### Initialize queue ###
        #
        queue: list[str] = []
        #
        ### Add initial node in queue ###
        #
        queue.append( src_node_id )
        #
        ### Exploration loop, explore while there are nodes in the queue ###
        #
        fn_on_loop_args_: list[Any] = fn_to_mark_nodes_args if fn_to_mark_nodes_args is not None else []
        fn_on_loop_kwargs_: dict[str, Any] = fn_to_mark_nodes_kwargs if fn_to_mark_nodes_kwargs is not None else {}
        #
        while queue and fn_on_loop(*fn_on_loop_args_, **fn_on_loop_kwargs_):
            #
            ### Get the next node to explore following the given algorithm ###
            #
            crt_node_id: str
            #
            if exploration_algorithm == "bfs":
                #
                crt_node_id = queue.pop(0)
            #
            elif exploration_algorithm == "dfs":
                #
                crt_node_id = queue.pop(-1)
            #
            else:
                #
                rid: int = random.randint(0, len(queue)-1)
                #
                crt_node_id = queue.pop(rid)
            #
            ### Check if the ode has already been marked ###
            #
            if crt_node_id in nodes_marks:
                #
                continue
            #
            ### Mark the node ###
            #
            fn_to_mark_nodes_args_: list[Any] = fn_to_mark_nodes_args if fn_to_mark_nodes_args is not None else []
            fn_to_mark_nodes_kwargs_: dict[str, Any] = fn_to_mark_nodes_kwargs if fn_to_mark_nodes_kwargs is not None else {}
            #
            nodes_marks[crt_node_id] = fn_to_mark_nodes( crt_node_id, *fn_to_mark_nodes_kwargs_, **fn_to_mark_nodes_kwargs_ )
            #
            ### Get neighbors ###
            #
            neighbors_ids: list[str] = self.get_successors_ids_of_node(node_id=crt_node_id)
            #
            if force_sym_edges_to_exists:
                #
                neighbors_ids += self.get_predecessors_ids_of_node(node_id=crt_node_id)
            #
            if custom_node_ordering is not None:
                #
                neighbors_ids.sort(keys=custom_node_ordering)
            #
            ### Explore neighbors ###
            #
            for neighbor_node_id in neighbors_ids:
                #
                ### Check if neighbor has been marked ###
                #
                if neighbor_node_id in nodes_marks:
                    #
                    continue
                #
                ### Add the neighbors to the queue ###
                #
                queue.append( neighbor_node_id )
        #
        ### Return curtom marked nodes, all the visited nodes have been marked, so nodes_marks.keys() is list of visited nodes ###
        #
        return nodes_marks

    #
    ### Function that performs a simple graph exploration and force all nodes exploration by calling auxiliar function until fully visited
    #
    def explore_all_nodes(
        self,
        exploration_algorithm: str = "bfs",
        force_sym_edges_to_exists: bool = False,
        custom_node_ordering: Optional[Callable[str, float]] = None,
        fn_to_mark_nodes: Callable[..., Any] = lambda _: 1,
        fn_to_mark_nodes_args: Optional[list[Any]] = None,
        fn_to_mark_nodes_kwargs: Optional[dict[str, Any]] = None,
        fn_on_loop: Callable[..., bool] = lambda _: True,
        fn_on_loop_args: Optional[list[Any]] = None,
        fn_on_loop_kwargs: Optional[dict[str, Any]] = None,
        fn_after_one_exploration: Callable[..., None] = lambda _: None,
        fn_after_one_exploration_args: Optional[list[Any]] = None,
        fn_after_one_exploration_kwargs: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        #
        ### Initialize marked nodes ###
        #
        nodes_marks: dict[str, Any] = {}
        #
        ### List to quickly access nodes that have not been visited ###
        #
        unvisited_nodes_lst: list[str] = list(self.nodes.keys())
        #
        if custom_node_ordering is not None:
            #
            unvisited_nodes_lst.sort(keys=custom_node_ordering)
        #
        unvisited_nodes: set[str] = set(unvisited_nodes_lst)
        #
        ### Wrapper function ###
        #
        def wrapper_fn_to_mark_nodes(visited_node_id: str, *args: list[Any], **kwargs: dict[str, Any]) -> Any:
            #
            ### Remove node from unvisited_nodes ###
            #
            unvisited_nodes.remove(visited_node_id)
            #
            ### call wrapped function ###
            #
            fn_on_loop(visited_node_id, *args, **kwargs)

        #
        fn_after_one_exploration_args_ = fn_after_one_exploration_args if fn_after_one_exploration_args is not None else []
        fn_after_one_exploration_kwargs_ = fn_after_one_exploration_kwargs if fn_after_one_exploration_kwargs is not None else {}

        #
        ### Explore ALL nodes ###
        #
        while unvisited_nodes:
            #
            new_source_node_id: str = next(iter(unvisited_nodes))
            #
            nodes_marks = self.explore_from_source(
                src_node_id = new_source_node_id,
                exploration_algorithm = exploration_algorithm,
                force_sym_edges_to_exists = force_sym_edges_to_exists,
                nodes_marks = nodes_marks,
                custom_node_ordering = custom_node_ordering,
                fn_to_mark_nodes = wrapper_fn_to_mark_nodes,
                fn_to_mark_nodes_args = fn_to_mark_nodes_args,
                fn_to_mark_nodes_kwargs = fn_to_mark_nodes_kwargs,
                fn_on_loop = fn_on_loop,
                fn_on_loop_args = fn_on_loop_args,
                fn_on_loop_kwargs = fn_on_loop_kwargs
            )
            #
            fn_after_one_exploration(*fn_after_one_exploration_args_, **fn_after_one_exploration_kwargs_)

        #
        ### Return marked nodes ###
        #
        return nodes_marks

    #
    ### Function to get all the connex composants of the graph ###
    #
    def get_all_connex_composants(
        self,
        force_sym_edges_to_exists: bool = False,
        custom_node_ordering: Optional[Callable[str, float]] = None
    ) -> tuple[dict[str, int], list[list[str]]]:
        #
        ### Init var ###
        #
        crt_composant: int = 0
        #
        ### Define mark function ###
        #
        def mark_composant(*args, **kwargs) -> int:
            #
            return crt_composant
        #
        ### Define increment function ###
        #
        def increment_composant(*args, **kwargs) -> None:
            #
            crt_composant += 1
        #
        composantes: dict[str, int] = self.explore_all_nodes(
            force_sym_edges_to_exists=force_sym_edges_to_exists,
            custom_node_ordering=custom_node_ordering,
            fn_to_mark_nodes=mark_composant,
            fn_after_one_exploration=increment_composant
        )
        #
        composantes_nodes_lst: dict[int, list[str]] = {}
        #
        for node_id, comp_id in composantes.items():
            #
            if comp_id not in composantes_nodes_lst:
                #
                composantes_nodes_lst[comp_id] = []
            #
            composantes_nodes_lst[comp_id].append( node_id )
        #
        composantes_nodes_lst_final: list[list[str]] = []
        #
        for i in range(len(composantes_nodes_lst)):
            #
            composantes_nodes_lst_final.append( composantes_nodes_lst[i] )
        #
        return composantes, composantes_nodes_lst_final


