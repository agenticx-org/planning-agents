import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional, Set, Any
# Removed: from sentence_transformers import SentenceTransformer # Use BERTopic's internal model
# from model2vec import StaticModel # Not used in this implementation, relying on BERTopic's embeddings
from bertopic import BERTopic
from datasets import load_dataset
from scipy.cluster import hierarchy as sch
import time
import logging
from collections import deque
from model2vec import StaticModel

class TreeNode:
    """Represents a node in the topic hierarchy."""
    def __init__(self, node_id: int, is_leaf: bool = False):
        self.node_id: int = node_id
        self.is_leaf: bool = is_leaf # True if it's a final topic ID from BERTopic
        self.children: List['TreeNode'] = [] # List of child TreeNode objects

    def add_child(self, child_node: 'TreeNode'):
        self.children.append(child_node)

    def __repr__(self):
        return f"TreeNode(id={self.node_id}, is_leaf={self.is_leaf}, children={len(self.children)})"


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SignalSearch():

    def __init__(self, load_local_model: bool = True, model_path: str = "old/bert_topic.pkl"):
        
        self.topic_model = BERTopic.load(model_path)
        self.embedding_model = self.topic_model.embedding_model
        self.embedding_model = StaticModel.from_pretrained("minishlab/M2V_base_output")
        dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
        self.docs = dataset['full']['text']
        logging.info(f"Dataset loaded with {len(self.docs)} documents.")

        logging.info("Calculating hierarchical topics...")
        # Use a linkage function suitable for your data/goals
        linkage_function = lambda x: sch.linkage(x, 'ward', optimal_ordering=True)
        self.hierarchical_df = self.topic_model.hierarchical_topics(self.docs, linkage_function=linkage_function)
        self.topic_df = self.topic_model.get_topic_info()


        # --- Precompute for efficiency ---
        # Create a mapping from topic ID to its embedding
        self.topic_embeddings_map = {
            topic_id: self.topic_model.topic_embeddings_[i]
            for i, topic_id in enumerate(self.topic_df['Topic'].values)
            if topic_id != -1 # Exclude outliers if necessary, or handle them
        }
        # Add embeddings for potential intermediate clusters if needed and available
        # Note: BERTopic's default topic_embeddings_ usually only contains final topics.
        # Handling intermediate node embeddings might require custom logic (e.g., averaging children).
        # For now, we will only score neighbors that are actual final topics.

        # Create a mapping from topic ID to list of document indices
        self.topic_to_doc_indices: Dict[int, List[int]] = {topic_id: [] for topic_id in self.topic_df['Topic']}
        for doc_idx, topic_id in enumerate(self.topic_model.topics_):
            if topic_id in self.topic_to_doc_indices:
                self.topic_to_doc_indices[topic_id].append(doc_idx)
        
        # --- Precompute for efficiency ---
        self.topic_embeddings_map = {
            topic_id: emb
            for topic_id, emb in zip(self.topic_df['Topic'], self.topic_model.topic_embeddings_) # Use zip for safety
            if topic_id != -1 and emb is not None # Exclude outliers and handle potential Nones
        }

        # Create a mapping from topic ID to list of document indices
        self.topic_to_doc_indices: Dict[int, List[int]] = {topic_id: [] for topic_id in self.topic_df['Topic']}
        for doc_idx, topic_id in enumerate(self.topic_model.topics_):
            if topic_id in self.topic_to_doc_indices:
                self.topic_to_doc_indices[topic_id].append(doc_idx)
        self.final_topic_ids = set(self.topic_to_doc_indices.keys()) # Set of actual leaf topic IDs (-1 included)
        logging.info("Precomputed mappings created.")



    def _embed_documents(self, documents: List[str]) -> np.ndarray:
        """Embeds a list of documents using the model's embedder."""
        # Add batching for potentially large inputs if needed
        return self.embedding_model.encode(documents, show_progress_bar=False) # , batch_size=128

    def query_n_topics(self, query: str, n_topics: int = 5) -> List[int]:
        """
        Finds the top N topics most similar to the query.

        Args:
            query (str): The user's search query.
            n_topics (int): The number of seed topics to return.

        Returns:
            List[int]: A list of the top N topic IDs. Returns empty list if query fails.
        """
        try:
            similar_topics, similarity = self.topic_model.find_topics(query, top_n=n_topics)
            logging.info(f"Query '{query}' found similar topics: {similar_topics} with similarities: {similarity}")
            # Filter out potential None or invalid topics
            valid_topics = [int(t) for t in similar_topics if isinstance(t, (int, np.integer)) and t >= -1]
            return valid_topics
        except Exception as e:
            logging.error(f"Error finding topics for query '{query}': {e}")
            return []

    def get_topic_docs(self, topic_id: int, max_docs: Optional[int] = None) -> List[str]:
        """
        Retrieves the actual documents associated with a specific final topic ID.

        Args:
            topic_id (int): The ID of the final topic (must exist in topic_to_doc_indices).
            max_docs (Optional[int]): Maximum number of documents to return.

        Returns:
            List[str]: A list of document texts. Empty list if invalid ID or no docs.
        """
        if topic_id not in self.topic_to_doc_indices:
             # This ID is either invalid or an intermediate cluster ID
             # logging.debug(f"Topic ID {topic_id} not found in final topic map. Skipping document retrieval.")
             return []

        doc_indices = self.topic_to_doc_indices[topic_id]
        if not doc_indices:
             return []

        if max_docs is not None and len(doc_indices) > max_docs:
            # Simple slicing for now, could be random sample
            doc_indices = doc_indices[:max_docs]

        try:
            # Ensure self.docs is list-like and indices are valid
            return [self.docs[i] for i in doc_indices if i < len(self.docs)]
        except IndexError:
            logging.error(f"Document index out of bounds retrieving docs for topic {topic_id}.")
            return []
        except Exception as e:
            logging.error(f"Error retrieving documents for topic {topic_id}: {e}")
            return []


    def get_topic_relationships(self, topic_id_to_find: int, debug: bool = False) -> List[int]:
        """
        Finds parent, children, sibling IDs for a given topic/cluster ID.
        Filters results to only include IDs that are actual final topics (leaf nodes).

        Args:
            topic_id_to_find (int): The ID of the topic or cluster to analyze.
            debug (bool): Whether to print debug information.

        Returns:
            List[int]: A list of valid related final topic IDs (parent's sibling if leaf, node's children if leaf).
                       Excludes intermediate cluster IDs. Returns neighbours that are final topics.
        """
        if not isinstance(self.hierarchical_df, pd.DataFrame):
            raise TypeError("hierarchical_df must be a Pandas DataFrame.")

        try:
            topic_id_to_find_int = int(topic_id_to_find)
        except (ValueError, TypeError):
             logging.warning(f"Invalid topic_id_to_find: {topic_id_to_find}. Must be integer-like.")
             return []

        related_ids_raw = set() # Use a set to avoid duplicates

        # Find row where topic_id is a child (to find parent and sibling)
        parent_row = self.hierarchical_df[
            (self.hierarchical_df['Child_Left_ID'] == topic_id_to_find_int) |
            (self.hierarchical_df['Child_Right_ID'] == topic_id_to_find_int)
        ]
        if not parent_row.empty:
            parent_row = parent_row.iloc[0]
            # Parent ID itself is usually intermediate, skip adding it directly
            # Find the sibling
            sibling_id = None
            if parent_row['Child_Left_ID'] == topic_id_to_find_int:
                sibling_id = parent_row['Child_Right_ID']
            else:
                sibling_id = parent_row['Child_Left_ID']

            if pd.notna(sibling_id): # Check if sibling exists and is not NaN
                 related_ids_raw.add(int(sibling_id))

        # Find row where topic_id is a parent (to find children)
        children_row = self.hierarchical_df[self.hierarchical_df['Parent_ID'] == topic_id_to_find_int]
        if not children_row.empty:
            children_row = children_row.iloc[0]
            child_left_id = children_row['Child_Left_ID']
            child_right_id = children_row['Child_Right_ID']
            if pd.notna(child_left_id):
                 related_ids_raw.add(int(child_left_id))
            if pd.notna(child_right_id):
                 related_ids_raw.add(int(child_right_id))

        # Filter raw IDs: Keep only those that are final topics (exist in self.final_topic_ids)
        final_related_ids = [rel_id for rel_id in related_ids_raw if rel_id in self.final_topic_ids]

        if debug:
            print(f"\n--- Debug: Relationships for Topic/Cluster {topic_id_to_find_int} ---")
            print(f"Raw Related IDs found (potential siblings/children): {related_ids_raw}")
            print(f"Final Topic IDs (leaves): {self.final_topic_ids}")
            print(f"Filtered Related IDs (actual topics): {final_related_ids}")
            print("--- End Debug ---")

        return final_related_ids


    def _calculate_novelty(self,
                          new_docs: List[str],
                          buffer_docs_embeddings: Optional[np.ndarray],
                          novelty_threshold: float = 0.8,
                          max_buffer_check: int = 500
                          ) -> Tuple[List[str], List[np.ndarray]]:
        """
        Checks which new documents are novel compared to existing buffer documents.
        Identical to the previous implementation.
        """
        if not new_docs:
            return [], []

        try:
            new_docs_embeddings = self._embed_documents(new_docs)
        except Exception as e:
            logging.error(f"Error embedding new documents for novelty check: {e}")
            return [], []


        if buffer_docs_embeddings is None or buffer_docs_embeddings.shape[0] == 0:
            return new_docs, list(new_docs_embeddings)

        # Limit buffer comparison for performance
        buffer_subset_embeddings = buffer_docs_embeddings
        if buffer_docs_embeddings.shape[0] > max_buffer_check:
             indices = np.random.choice(buffer_docs_embeddings.shape[0], max_buffer_check, replace=False)
             buffer_subset_embeddings = buffer_docs_embeddings[indices]

        try:
            # Calculate similarity: rows=new_docs, cols=buffer_docs
            similarity_matrix = cosine_similarity(new_docs_embeddings, buffer_subset_embeddings)
        except Exception as e:
             logging.error(f"Error calculating cosine similarity for novelty: {e}")
             # Handle potential dimension mismatch or other errors
             return [], [] # Return no novel docs if similarity fails

        # Find max similarity for each new doc against the buffer subset
        max_similarities = np.max(similarity_matrix, axis=1)

        novel_indices = np.where(max_similarities <= novelty_threshold)[0]

        novel_documents = [new_docs[i] for i in novel_indices]
        novel_embeddings = [new_docs_embeddings[i] for i in novel_indices]

        return novel_documents, novel_embeddings


    def crawl(self,
              query: str,
              n_seed_topics: int = 5,
              max_steps: int = 20,
              novelty_threshold: float = 0.8,
              stop_patience: int = 3,
              min_new_docs_rate: float = 1.0, # Allow float for avg
              max_docs_per_topic: int = 50,
              max_buffer_check: int = 500
              ) -> Dict[int, List[str]]:
        """
        Performs focused crawling starting from query-similar topics.
        (Implementation largely unchanged from previous version)
        """
        logging.info(f"Starting query-focused crawl for: '{query}'")
        seed_topics = self.query_n_topics(query, n_topics=n_seed_topics)
        if not seed_topics:
            logging.warning("No seed topics found for the query. Stopping crawl.")
            return {}

        try:
            query_embedding = self._embed_documents([query])[0]
        except Exception as e:
            logging.error(f"Failed to embed query '{query}': {e}")
            return {}

        # --- State Initialization ---
        document_buffers: Dict[int, List[str]] = {seed: [] for seed in seed_topics}
        buffer_embeddings: Dict[int, Optional[np.ndarray]] = {seed: None for seed in seed_topics}
        visited_nodes: Dict[int, Set[int]] = {seed: set() for seed in seed_topics}
        frontiers: Dict[int, List[Tuple[int, float]]] = {seed: [] for seed in seed_topics}
        collection_history: Dict[int, deque[int]] = {seed: deque(maxlen=stop_patience) for seed in seed_topics}
        consecutive_low_collection: Dict[int, int] = {seed: 0 for seed in seed_topics}
        active_seeds = set(seed_topics)
        current_nodes: Dict[int, int] = {seed: seed for seed in seed_topics}

        # Initialize buffers
        for seed in seed_topics:
            logging.info(f"Initializing seed topic {seed}...")
            initial_docs = self.get_topic_docs(seed, max_docs=max_docs_per_topic)
            if initial_docs:
                 # Use novelty check even for initial docs (though buffer is empty)
                 novel_docs, novel_embeds = self._calculate_novelty(initial_docs, None, novelty_threshold)
                 if novel_docs:
                     document_buffers[seed].extend(novel_docs)
                     buffer_embeddings[seed] = np.array(novel_embeds)
                     logging.info(f"Seed {seed}: Added {len(novel_docs)} initial documents.")
                     collection_history[seed].append(len(novel_docs))
                 else:
                      collection_history[seed].append(0) # Handles case where initial docs somehow fail novelty
            else:
                 collection_history[seed].append(0)
                 # logging.warning(f"Seed topic {seed} has no initial documents.") # Less noisy
            visited_nodes[seed].add(seed)

        # --- Crawling Loop ---
        for step in range(max_steps):
            if not active_seeds:
                logging.info("All seed paths have stopped. Ending query crawl.")
                break

            logging.info(f"\n--- Query Crawl Step {step + 1} / {max_steps} --- (Active Seeds: {len(active_seeds)})")
            seeds_to_deactivate = set()

            for seed_id in list(active_seeds):
                current_node_id = current_nodes[seed_id]
                logging.debug(f"Query Crawl: Seed Path {seed_id}, Current Node: {current_node_id}")

                # 1. Find and Score Neighbors (only if frontier empty)
                if not frontiers[seed_id]:
                    # Get *final topic* neighbors based on hierarchy structure around current node
                    neighbors = self.get_topic_relationships(current_node_id)
                    candidate_neighbors = []
                    for neighbor_id in neighbors:
                        if neighbor_id != -1 and neighbor_id not in visited_nodes[seed_id]: # Exclude outliers, check visited for *this path*
                            neighbor_embedding = self.topic_embeddings_map.get(neighbor_id)
                            if neighbor_embedding is not None:
                                try:
                                    score = cosine_similarity(query_embedding.reshape(1, -1), neighbor_embedding.reshape(1, -1))[0][0]
                                    candidate_neighbors.append((neighbor_id, score))
                                except Exception as score_e:
                                     logging.warning(f"Could not calculate score for neighbor {neighbor_id}: {score_e}")
                            # else: logging.debug(f"No embedding found for neighbor {neighbor_id}.")

                    candidate_neighbors.sort(key=lambda x: x[1], reverse=True) # Higher score = more similar to query
                    frontiers[seed_id] = candidate_neighbors
                    logging.debug(f"Query Crawl Seed {seed_id}: Found {len(candidate_neighbors)} new candidate neighbors for node {current_node_id}.")

                # 2. Select Next Node from Frontier
                if not frontiers[seed_id]:
                    logging.debug(f"Query Crawl Seed {seed_id}: No valid, unvisited neighbors from node {current_node_id}.")
                    collection_history[seed_id].append(0) # No move, no collection
                else:
                    next_node_id, score = frontiers[seed_id].pop(0)
                    logging.info(f"Query Crawl Seed {seed_id}: Moving from {current_node_id} -> {next_node_id} (Query Sim: {score:.4f})")
                    current_nodes[seed_id] = next_node_id
                    visited_nodes[seed_id].add(next_node_id) # Mark visited for this path

                    # 3. Fetch & Check Novelty for the New Node
                    new_docs = self.get_topic_docs(next_node_id, max_docs=max_docs_per_topic)
                    num_added = 0
                    if new_docs:
                        novel_documents, novel_embeddings = self._calculate_novelty(
                            new_docs,
                            buffer_embeddings[seed_id], # Compare against this path's buffer
                            novelty_threshold,
                            max_buffer_check
                        )
                        if novel_documents:
                            num_added = len(novel_documents)
                            logging.info(f"Query Crawl Seed {seed_id}, Node {next_node_id}: Added {num_added} novel documents.")
                            document_buffers[seed_id].extend(novel_documents)
                            if buffer_embeddings[seed_id] is None:
                                buffer_embeddings[seed_id] = np.array(novel_embeddings)
                            else:
                                buffer_embeddings[seed_id] = np.vstack([buffer_embeddings[seed_id], np.array(novel_embeddings)])
                        # else: logging.debug(f"Query Crawl Seed {seed_id}, Node {next_node_id}: Docs found but none novel.")
                    # else: logging.debug(f"Query Crawl Seed {seed_id}, Node {next_node_id}: No documents found.")
                    collection_history[seed_id].append(num_added) # Track how many were added this step

                # 5. Check Stopping Condition
                if len(collection_history[seed_id]) >= stop_patience: # Use >= for safety
                    # Check average over the window
                    avg_recent_collection = sum(collection_history[seed_id]) / len(collection_history[seed_id])
                    if avg_recent_collection < min_new_docs_rate:
                         consecutive_low_collection[seed_id] += 1
                         logging.debug(f"Query Crawl Seed {seed_id}: Low collection rate (Avg: {avg_recent_collection:.2f} < {min_new_docs_rate}). Patience: {consecutive_low_collection[seed_id]}/{stop_patience}")
                         if consecutive_low_collection[seed_id] >= stop_patience:
                             seeds_to_deactivate.add(seed_id)
                             logging.info(f"--- Stopping query crawl for Seed Path {seed_id} due to low collection rate. ---")
                    else:
                         consecutive_low_collection[seed_id] = 0 # Reset patience

            active_seeds -= seeds_to_deactivate
            time.sleep(0.05) # Small delay

        logging.info(f"\n--- Query Crawl Finished ---")
        total_docs = sum(len(docs) for docs in document_buffers.values())
        logging.info(f"Collected {total_docs} novel documents via query crawl across {len(seed_topics)} seed paths.")
        return document_buffers

    # --- NEW METHOD ---
        # --- METHOD DEFINITION ---
        # In class SignalSearch:

    def _build_hierarchy_tree(self) -> Optional[TreeNode]:
        """
        Parses the hierarchical_df and builds an explicit tree structure.

        Returns:
            Optional[TreeNode]: The root TreeNode of the hierarchy, or None if build fails.
        """
        logging.info("Building explicit hierarchy tree from hierarchical_df...")
        if not isinstance(self.hierarchical_df, pd.DataFrame) or self.hierarchical_df.empty:
            logging.error("Cannot build tree: hierarchical_df is missing or empty.")
            return None
        if not hasattr(self, 'final_topic_ids') or not self.final_topic_ids:
             logging.error("Cannot build tree: final_topic_ids attribute is missing or empty.")
             return None

        required_cols = {'Parent_ID', 'Child_Left_ID', 'Child_Right_ID'}
        if not required_cols.issubset(self.hierarchical_df.columns):
            logging.error(f"Cannot build tree: hierarchical_df missing columns: {required_cols - set(self.hierarchical_df.columns)}")
            return None

        # --- 1. Create all nodes ---
        nodes_map: Dict[int, TreeNode] = {}
        all_ids = set()
        try:
            # Collect all unique valid integer IDs from the hierarchy table
            for col in ['Parent_ID', 'Child_Left_ID', 'Child_Right_ID']:
                # Convert to numeric, drop NaN, convert to int, add to set
                valid_ids = pd.to_numeric(self.hierarchical_df[col], errors='coerce').dropna().astype(int).unique()
                all_ids.update(valid_ids)

            if not all_ids:
                 logging.error("No valid numeric IDs found in hierarchical_df columns.")
                 return None

            # Create TreeNode objects for all identified IDs
            for node_id in all_ids:
                is_leaf_node = node_id in self.final_topic_ids
                nodes_map[node_id] = TreeNode(node_id=node_id, is_leaf=is_leaf_node)
            logging.info(f"Created {len(nodes_map)} TreeNode objects initially.")

        except Exception as e:
            logging.error(f"Error during node creation phase: {e}")
            return None

        # --- 2. Link children to parents ---
        try:
            for _, row in self.hierarchical_df.iterrows():
                parent_id_raw = row.get('Parent_ID')
                child_left_id_raw = row.get('Child_Left_ID')
                child_right_id_raw = row.get('Child_Right_ID')

                # Convert IDs to int, skip if invalid or NaN
                parent_id = pd.to_numeric(parent_id_raw, errors='coerce')
                child_left_id = pd.to_numeric(child_left_id_raw, errors='coerce')
                child_right_id = pd.to_numeric(child_right_id_raw, errors='coerce')

                if pd.isna(parent_id): continue # Skip rows with invalid parent ID

                parent_id = int(parent_id)
                parent_node = nodes_map.get(parent_id)

                if parent_node is None:
                    # This shouldn't happen if all_ids collection was correct, but check anyway
                    logging.warning(f"Parent ID {parent_id} from row not found in nodes_map. Skipping row.")
                    continue

                # Link Left Child
                if pd.notna(child_left_id):
                    child_left_id = int(child_left_id)
                    child_node = nodes_map.get(child_left_id)
                    if child_node:
                        parent_node.add_child(child_node)
                    else:
                        logging.warning(f"Child_Left ID {child_left_id} from row not found in nodes_map.")

                # Link Right Child
                if pd.notna(child_right_id):
                    child_right_id = int(child_right_id)
                    child_node = nodes_map.get(child_right_id)
                    if child_node:
                        parent_node.add_child(child_node)
                    else:
                        logging.warning(f"Child_Right ID {child_right_id} from row not found in nodes_map.")

        except Exception as e:
            logging.error(f"Error during node linking phase: {e}")
            return None


        # --- 3. Find the root node ---
        try:
            parent_ids = set(pd.to_numeric(self.hierarchical_df['Parent_ID'], errors='coerce').dropna().astype(int))
            child_ids = set()
            if 'Child_Left_ID' in self.hierarchical_df.columns:
                 child_ids.update(pd.to_numeric(self.hierarchical_df['Child_Left_ID'], errors='coerce').dropna().astype(int))
            if 'Child_Right_ID' in self.hierarchical_df.columns:
                 child_ids.update(pd.to_numeric(self.hierarchical_df['Child_Right_ID'], errors='coerce').dropna().astype(int))

            potential_roots = parent_ids - child_ids

            if not potential_roots:
                if parent_ids:
                    root_id = max(parent_ids) # Fallback to max parent ID
                    logging.warning(f"Could not identify unique root node (never a child). Using highest Parent_ID {root_id} as root.")
                else:
                    logging.error("No valid parent IDs found. Cannot determine root.")
                    return None
            elif len(potential_roots) > 1:
                 root_id = max(potential_roots) # Take highest if multiple roots
                 logging.warning(f"Multiple potential root nodes found ({potential_roots}). Using highest ID {root_id} as root.")
            else:
                 root_id = potential_roots.pop()

            root_node = nodes_map.get(root_id)
            if root_node:
                logging.info(f"Hierarchy tree built successfully. Root: {root_node}")
                return root_node
            else:
                logging.error(f"Determined root ID {root_id} not found in nodes_map.")
                return None

        except Exception as e:
            logging.error(f"Error during root finding phase: {e}")
            return None

    # --- REWRITTEN crawl_hierarchical ---
    def crawl_hierarchical(self,
                           max_nodes: int = 100,
                           max_docs_collected: int = 1000,
                           novelty_threshold: float = 0.85,
                           stop_patience: int = 5,
                           min_novel_docs_per_leaf: int = 1,
                           max_docs_per_topic: int = 20,
                           max_buffer_check: int = 500,
                           traversal_mode: str = 'bfs' # 'bfs' or 'dfs'
                           ) -> List[str]:
        """
        Crawls the topic hierarchy using a pre-built tree structure
        to collect a representative set of documents.
        """
        logging.info(f"Starting hierarchical crawl using pre-built tree (Mode: {traversal_mode}, Max Nodes: {max_nodes}, Max Docs: {max_docs_collected})")

        # --- 1. Build the Tree ---
        root_node = self._build_hierarchy_tree()
        if root_node is None:
            logging.error("Failed to build hierarchy tree. Stopping hierarchical crawl.")
            return []

        # --- 2. State Initialization ---
        global_document_buffer: List[str] = []
        global_buffer_embeddings: Optional[np.ndarray] = None
        visited_node_ids: Set[int] = set() # Track visited node IDs
        frontier = deque() # Queue for BFS/DFS: stores (TreeNode, depth)
        consecutive_low_novelty_leaves = 0 # Counter for stopping condition
        nodes_visited_count = 0

        # Add root to frontier
        frontier.append((root_node, 0))

        # --- 3. Crawling Loop (BFS/DFS on TreeNodes) ---
        while frontier:
            # Check global stopping conditions first
            if nodes_visited_count >= max_nodes:
                logging.info(f"Stopping hierarchical crawl: Maximum nodes visited ({max_nodes}).")
                break
            if len(global_document_buffer) >= max_docs_collected:
                logging.info(f"Stopping hierarchical crawl: Maximum documents collected ({max_docs_collected}).")
                break
            if consecutive_low_novelty_leaves >= stop_patience:
                 logging.info(f"Stopping hierarchical crawl: Novelty stagnated for {stop_patience} consecutive leaf nodes.")
                 break

            # Get next node from frontier based on traversal mode
            try:
                if traversal_mode == 'bfs':
                    current_treenode, current_depth = frontier.popleft()
                elif traversal_mode == 'dfs':
                    current_treenode, current_depth = frontier.pop()
                else:
                    logging.error(f"Invalid traversal_mode: {traversal_mode}. Use 'bfs' or 'dfs'.")
                    break
            except IndexError:
                 logging.info("Frontier is empty. Hierarchical crawl finished.")
                 break # Exit loop if frontier is empty

            current_node_id = current_treenode.node_id

            # Skip if already visited
            if current_node_id in visited_node_ids:
                continue

            # Process node
            visited_node_ids.add(current_node_id)
            nodes_visited_count += 1
            logging.debug(f"Hierarchical Crawl: Visiting {current_treenode} (Depth: {current_depth}, Total Visited: {nodes_visited_count})")

            # Check if it's a leaf node (a final topic)
            if current_treenode.is_leaf:
                # --- Leaf Node Processing ---
                logging.debug(f"Node {current_node_id} is a final topic (leaf). Fetching documents.")
                topic_docs = self.get_topic_docs(current_node_id, max_docs=max_docs_per_topic)
                num_added = 0
                if topic_docs:
                    novel_documents, novel_embeddings = self._calculate_novelty(
                        topic_docs,
                        global_buffer_embeddings, # Compare against the global buffer
                        novelty_threshold,
                        max_buffer_check
                    )
                    if novel_documents:
                        num_added = len(novel_documents)
                        global_document_buffer.extend(novel_documents)
                        # Update global embeddings (with safety checks)
                        if global_buffer_embeddings is None:
                           if novel_embeddings: # Check if novel_embeddings is not empty
                               global_buffer_embeddings = np.array(novel_embeddings)
                        elif novel_embeddings: # Check if novel_embeddings is not empty
                           new_embeds_arr = np.array(novel_embeddings)
                           # Ensure 2D for vstack
                           if new_embeds_arr.ndim == 1: new_embeds_arr = new_embeds_arr.reshape(1, -1)
                           if global_buffer_embeddings.ndim == 1: global_buffer_embeddings = global_buffer_embeddings.reshape(1, -1)

                           if new_embeds_arr.size > 0:
                               if new_embeds_arr.shape[1] != global_buffer_embeddings.shape[1]:
                                   logging.error(f"Embedding dimension mismatch! Buffer: {global_buffer_embeddings.shape}, New: {new_embeds_arr.shape}. Skipping append.")
                               else:
                                   try:
                                       global_buffer_embeddings = np.vstack([global_buffer_embeddings, new_embeds_arr])
                                   except ValueError as ve:
                                        logging.error(f"Error during vstack of embeddings: {ve}. Buffer shape: {global_buffer_embeddings.shape}, New shape: {new_embeds_arr.shape}")

                        if num_added > 0:
                             logging.info(f"Hierarchical Crawl: Added {num_added} novel documents from Topic {current_node_id}. Total: {len(global_document_buffer)}")
                    # else: logging.debug(f"Topic {current_node_id}: Documents checked, none novel.")
                # else: logging.debug(f"Topic {current_node_id}: No documents found.")

                # Update stagnation counter based on this leaf node's contribution
                if num_added < min_novel_docs_per_leaf:
                    consecutive_low_novelty_leaves += 1
                    logging.debug(f"Low novelty from leaf {current_node_id} ({num_added} < {min_novel_docs_per_leaf}). Patience: {consecutive_low_novelty_leaves}/{stop_patience}")
                else:
                    consecutive_low_novelty_leaves = 0 # Reset patience

            else:
                # --- Intermediate Node Processing ---
                # Add its children (which are TreeNode objects) to the frontier
                children_added_count = 0
                for child_node in current_treenode.children:
                    if child_node.node_id not in visited_node_ids:
                        frontier.append((child_node, current_depth + 1))
                        children_added_count += 1

                if children_added_count > 0:
                    logging.debug(f"Added {children_added_count} children of node {current_node_id} to frontier.")
                # else: logging.debug(f"Node {current_node_id} is intermediate, but has no *new* children to add to frontier.")


            logging.debug(f"End of iteration for Node {current_node_id}. Frontier size: {len(frontier)}")
            time.sleep(0.01) # Small delay

        logging.info(f"\n--- Hierarchical Crawl Finished ---")
        logging.info(f"Visited {nodes_visited_count} nodes.")
        logging.info(f"Collected a total of {len(global_document_buffer)} novel documents globally.")

        return global_document_buffer


# --- Example Usage ---
if __name__ == "__main__":
    # Initialize the search object
    # Set load_local_model=False ONLY if you need to train (will take time/memory!)
    # Ensure the path "old/bert_topic.pkl" exists if load_local_model=True
    try:
        # Increase memory or use smaller subset if running into issues on full data
        signal_search_instance = SignalSearch(load_local_model=True, model_path="old/bert_topic.pkl")

        # --- Optional: Test Query-Based Crawl ---
        # print("\n--- Starting Query Crawl ---")
        # user_query = "anti aging eye serum"
        # query_collected_data = signal_search_instance.crawl(
        #     query=user_query,
        #     n_seed_topics=5,
        #     max_steps=10,
        #     novelty_threshold=0.75,
        #     stop_patience=3,
        #     min_new_docs_rate=1,
        #     max_docs_per_topic=15,
        #     max_buffer_check=300
        # )
        # print("\n--- Query Crawl Results ---")
        # # ... (print results summary)


        # --- Run the Hierarchical Crawl ---
        print("\n--- Starting Hierarchical Crawl (BFS) ---")
        representative_docs_bfs = signal_search_instance.crawl_hierarchical(
            max_nodes=150,                 # Visit up to 150 nodes (clusters + topics)
            max_docs_collected=500,       # Stop after collecting 500 novel docs
            novelty_threshold=0.80,       # Similarity threshold for novelty
            stop_patience=8,              # Stop if 8 consecutive leaf nodes add < min_novel_docs
            min_novel_docs_per_leaf=2,    # Need at least 2 novel docs from a leaf to reset patience
            max_docs_per_topic=10,        # Check max 10 docs per leaf topic
            max_buffer_check=400,
            traversal_mode='bfs'          # Explore level-by-level
        )

        print("\n--- Hierarchical Crawl Results (BFS) ---")
        print(f"Collected {len(representative_docs_bfs)} representative documents.")
        # Optionally print some snippets
        if representative_docs_bfs:
             print("Examples (first 5):")
             for i, doc in enumerate(representative_docs_bfs[:5]):
                 print(f"  - {doc[:150]}...")


        # --- Optional: Run Hierarchical Crawl (DFS) ---
        # print("\n--- Starting Hierarchical Crawl (DFS) ---")
        # representative_docs_dfs = signal_search_instance.crawl_hierarchical(
        #     max_nodes=150,
        #     max_docs_collected=500,
        #     novelty_threshold=0.80,
        #     stop_patience=8,
        #     min_novel_docs_per_leaf=2,
        #     max_docs_per_topic=10,
        #     max_buffer_check=400,
        #     traversal_mode='dfs' # Explore depth-first
        # )
        # print("\n--- Hierarchical Crawl Results (DFS) ---")
        # print(f"Collected {len(representative_docs_dfs)} representative documents.")
        # # ... (print examples)


    except AttributeError as ae:
         logging.error(f"Attribute Error: {ae}. This might indicate issues with the loaded model or data structures.")
    except MemoryError:
         logging.error("Memory Error occurred. Consider using a smaller dataset split, reducing batch sizes, or increasing available RAM.")
    except Exception as e:
        logging.exception(f"An unexpected error occurred during execution: {e}")