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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SignalSearch(): # Renamed class slightly to follow convention

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
        

    def _embed_documents(self, documents: List[str]) -> np.ndarray:
        """Embeds a list of documents using the model's embedder."""
        return self.embedding_model.encode(documents, show_progress_bar=False)

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
            # Filter out potential None or invalid topics if find_topics behaves unexpectedly
            valid_topics = [t for t in similar_topics if isinstance(t, (int, np.integer)) and t >= -1] # Allow -1 (outliers) if needed, though usually skipped
            return valid_topics
        except Exception as e:
            logging.error(f"Error finding topics for query '{query}': {e}")
            return []

    def get_topic_docs(self, topic_id: int, max_docs: Optional[int] = None) -> List[str]:
        """
        Retrieves the actual documents associated with a specific topic ID.

        Args:
            topic_id (int): The ID of the topic.
            max_docs (Optional[int]): Maximum number of documents to return for the topic.
                                      Returns all if None.

        Returns:
            List[str]: A list of document texts for the topic.
                       Returns empty list if topic ID not found or has no documents.
        """
        if topic_id not in self.topic_to_doc_indices:
            # This can happen if the ID is an intermediate cluster ID not in topic_df
            # Or if the topic ID is invalid
            # logging.warning(f"Topic ID {topic_id} not found in precomputed doc map. It might be an intermediate cluster or invalid.")
             return [] # Return empty list for intermediate nodes for now

        doc_indices = self.topic_to_doc_indices[topic_id]
        if not doc_indices:
             return []

        if max_docs is not None and len(doc_indices) > max_docs:
            # Optional: Select representative or random subset if max_docs is specified
            # Here, we just take the first 'max_docs' for simplicity
            doc_indices = doc_indices[:max_docs]

        # Retrieve documents using indices (ensure self.docs is accessible)
        try:
            return [self.docs[i] for i in doc_indices]
        except IndexError:
            logging.error(f"Document index out of bounds for topic {topic_id}. Data inconsistency?")
            return []
        except Exception as e:
            logging.error(f"Error retrieving documents for topic {topic_id}: {e}")
            return []


    def get_topic_relationships(self, topic_id_to_find: int, debug: bool = False) -> List[int]:
        """
        Finds parent, children, sibling IDs for a given topic ID within the hierarchy.
        Filters results to only include valid topic IDs present in the main topic_df.

        Args:
            topic_id_to_find (int): The ID of the topic or cluster to analyze.
            debug (bool): Whether to print debug information.

        Returns:
            List[int]: A list of valid related topic IDs (parent, children, sibling).
                       Excludes intermediate cluster IDs not found in self.topic_df.
        """
        # --- [Code from the original prompt for finding relationships] ---
        # This internal logic remains the same as provided in the prompt
        # ... (finding parent_merge_row, children_merge_row, cousins_merge_row etc.) ...
        # --- [End of original prompt code section] ---

        # --- Start Copied Logic from Prompt (with minor adjustments) ---
        if not isinstance(self.hierarchical_df, pd.DataFrame):
            raise TypeError("hierarchical_df must be a Pandas DataFrame.")
        required_cols = ['Parent_ID', 'Parent_Name', 'Child_Left_ID', 'Child_Left_Name',
                        'Child_Right_ID', 'Child_Right_Name']
        if not all(col in self.hierarchical_df.columns for col in required_cols):
            raise ValueError(f"hierarchical_df is missing one or more required columns: {required_cols}")

        # Ensure hierarchical DF IDs are numeric (handle potential loading issues)
        for col in ['Parent_ID', 'Child_Left_ID', 'Child_Right_ID']:
             if col in self.hierarchical_df.columns:
                 self.hierarchical_df[col] = pd.to_numeric(self.hierarchical_df[col], errors='coerce')

        try:
            topic_id_to_find = int(topic_id_to_find)
        except (ValueError, TypeError):
             logging.warning(f"Invalid topic_id_to_find: {topic_id_to_find}. Must be integer-like.")
             return []

        parent_id: Optional[int] = None
        sibling_id: Optional[int] = None
        children_ids: List[int] = []
        # Cousins are ignored in this version for tighter focus, but could be added back

        # --- 1. Find Parent and Sibling ---
        parent_merge_row = self.hierarchical_df[
            (self.hierarchical_df['Child_Left_ID'] == topic_id_to_find) |
            (self.hierarchical_df['Child_Right_ID'] == topic_id_to_find)
        ]

        if not parent_merge_row.empty:
            parent_merge_row = parent_merge_row.iloc[0]
            parent_id = int(parent_merge_row['Parent_ID']) if pd.notna(parent_merge_row['Parent_ID']) else None

            if parent_merge_row['Child_Left_ID'] == topic_id_to_find:
                sibling_id = int(parent_merge_row['Child_Right_ID']) if pd.notna(parent_merge_row['Child_Right_ID']) else None
            else:
                sibling_id = int(parent_merge_row['Child_Left_ID']) if pd.notna(parent_merge_row['Child_Left_ID']) else None

        # --- 2. Find Children ---
        children_merge_row = self.hierarchical_df[self.hierarchical_df['Parent_ID'] == topic_id_to_find]

        if not children_merge_row.empty:
            children_merge_row = children_merge_row.iloc[0]
            child_left_id = int(children_merge_row['Child_Left_ID']) if pd.notna(children_merge_row['Child_Left_ID']) else None
            child_right_id = int(children_merge_row['Child_Right_ID']) if pd.notna(children_merge_row['Child_Right_ID']) else None
            if child_left_id is not None:
                children_ids.append(child_left_id)
            if child_right_id is not None:
                children_ids.append(child_right_id)

        # --- Combine and Filter ---
        related_ids_raw: List[int] = []
        if parent_id is not None:
            related_ids_raw.append(parent_id)
        if sibling_id is not None:
            related_ids_raw.append(sibling_id)
        related_ids_raw.extend(children_ids)

        # Filter to only include IDs that are actual topics (in topic_df) or -1
        # This prevents trying to fetch docs/embeddings for intermediate cluster IDs directly
        valid_topic_ids = set(self.topic_df['Topic'].unique())
        final_related_ids = [rel_id for rel_id in related_ids_raw if rel_id in valid_topic_ids]

        if debug:
            print(f"\n--- Debug: Relationships for Topic {topic_id_to_find} ---")
            print(f"Raw Parent ID: {parent_id}")
            print(f"Raw Sibling ID: {sibling_id}")
            print(f"Raw Children IDs: {children_ids}")
            print(f"Raw Related IDs: {related_ids_raw}")
            print(f"Valid Final Topic IDs: {valid_topic_ids}")
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

        Args:
            new_docs (List[str]): List of new document texts to check.
            buffer_docs_embeddings (Optional[np.ndarray]): Embeddings of documents already in the buffer.
                                                          None if buffer is empty.
            novelty_threshold (float): Cosine similarity threshold. Docs with max similarity
                                       *above* this are considered NOT novel.
            max_buffer_check (int): Max number of buffer embeddings to compare against for efficiency.

        Returns:
            Tuple[List[str], List[np.ndarray]]:
                - List of novel document texts.
                - List of embeddings for the novel documents.
        """
        if not new_docs:
            return [], []

        new_docs_embeddings = self._embed_documents(new_docs)

        if buffer_docs_embeddings is None or buffer_docs_embeddings.shape[0] == 0:
            # If buffer is empty, all new docs are novel
            return new_docs, list(new_docs_embeddings) # Return embeddings as list for easier appending

        # Limit buffer comparison for performance
        if buffer_docs_embeddings.shape[0] > max_buffer_check:
             # Sample or select the most recent/representative embeddings
             indices = np.random.choice(buffer_docs_embeddings.shape[0], max_buffer_check, replace=False)
             buffer_subset_embeddings = buffer_docs_embeddings[indices]
        else:
             buffer_subset_embeddings = buffer_docs_embeddings

        # Calculate similarity: rows=new_docs, cols=buffer_docs
        similarity_matrix = cosine_similarity(new_docs_embeddings, buffer_subset_embeddings)

        # Find max similarity for each new doc against the buffer subset
        max_similarities = np.max(similarity_matrix, axis=1)

        novel_indices = np.where(max_similarities <= novelty_threshold)[0]

        novel_documents = [new_docs[i] for i in novel_indices]
        novel_embeddings = [new_docs_embeddings[i] for i in novel_indices]

        # logging.debug(f"Novelty check: {len(novel_documents)} / {len(new_docs)} documents are novel (threshold {novelty_threshold}).")
        return novel_documents, novel_embeddings


    def crawl(self,
              query: str,
              n_seed_topics: int = 5,
              max_steps: int = 20,
              novelty_threshold: float = 0.8,
              stop_patience: int = 3,
              min_new_docs_rate: int = 1,
              max_docs_per_topic: int = 50,
              max_buffer_check: int = 500
              ) -> Dict[int, List[str]]:
        """
        Performs the focused crawling process.

        Args:
            query (str): The user's initial query.
            n_seed_topics (int): Number of initial seed topics to start from.
            max_steps (int): Maximum number of exploration steps per seed path.
            novelty_threshold (float): Cosine similarity threshold for document novelty.
            stop_patience (int): Number of consecutive steps with low collection rate before stopping a path.
            min_new_docs_rate (int): Minimum number of new documents to add per step to be considered active collecting.
            max_docs_per_topic (int): Max documents to fetch and check from each visited topic node.
            max_buffer_check (int): Max buffer documents to compare against for novelty check efficiency.

        Returns:
            Dict[int, List[str]]: A dictionary where keys are seed topic IDs and values
                                  are lists of collected novel document texts for that path.
        """
        logging.info(f"Starting crawl for query: '{query}'")
        seed_topics = self.query_n_topics(query, n_topics=n_seed_topics)
        if not seed_topics:
            logging.warning("No seed topics found for the query. Stopping crawl.")
            return {}

        query_embedding = self._embed_documents([query])[0]

        # --- State Initialization ---
        document_buffers: Dict[int, List[str]] = {seed: [] for seed in seed_topics}
        buffer_embeddings: Dict[int, Optional[np.ndarray]] = {seed: None for seed in seed_topics} # Store embeddings for novelty checks
        visited_nodes: Dict[int, Set[int]] = {seed: set() for seed in seed_topics} # Track visited nodes per seed path
        # Track candidate nodes (topic_id, score) to explore for each seed path
        frontiers: Dict[int, List[Tuple[int, float]]] = {seed: [] for seed in seed_topics}
        # Track the number of new docs added in recent steps for stopping condition
        collection_history: Dict[int, deque[int]] = {seed: deque(maxlen=stop_patience) for seed in seed_topics}
        consecutive_low_collection: Dict[int, int] = {seed: 0 for seed in seed_topics}
        active_seeds = set(seed_topics) # Seeds currently being explored
        current_nodes: Dict[int, int] = {seed: seed for seed in seed_topics} # Node being explored for each seed

        # Initialize buffers with seed topic documents
        for seed in seed_topics:
            logging.info(f"Initializing seed topic {seed}...")
            initial_docs = self.get_topic_docs(seed, max_docs=max_docs_per_topic)
            if initial_docs:
                 novel_docs, novel_embeds = self._calculate_novelty(initial_docs, None, novelty_threshold) # Initially buffer is empty
                 if novel_docs:
                     document_buffers[seed].extend(novel_docs)
                     buffer_embeddings[seed] = np.array(novel_embeds) # Initialize embeddings
                     logging.info(f"Seed {seed}: Added {len(novel_docs)} initial documents.")
                     collection_history[seed].append(len(novel_docs))
                 else:
                     collection_history[seed].append(0)
            else:
                 collection_history[seed].append(0)
                 logging.warning(f"Seed topic {seed} has no initial documents.")
            visited_nodes[seed].add(seed) # Mark seed as visited for its path


        # --- Crawling Loop ---
        for step in range(max_steps):
            if not active_seeds:
                logging.info("All seed paths have stopped. Ending crawl.")
                break

            logging.info(f"\n--- Step {step + 1} / {max_steps} --- (Active Seeds: {len(active_seeds)})")
            seeds_to_deactivate = set()

            for seed_id in list(active_seeds): # Iterate over a copy as we might modify active_seeds
                current_node_id = current_nodes[seed_id]
                logging.debug(f"Processing Seed Path {seed_id}, Current Node: {current_node_id}")

                # 1. Find and Score Neighbors (if frontier is empty)
                if not frontiers[seed_id]:
                    neighbors = self.get_topic_relationships(current_node_id)
                    candidate_neighbors = []
                    for neighbor_id in neighbors:
                        if neighbor_id not in visited_nodes[seed_id]:
                            # Score neighbor based on similarity to the *original query*
                            neighbor_embedding = self.topic_embeddings_map.get(neighbor_id)
                            if neighbor_embedding is not None:
                                score = cosine_similarity(query_embedding.reshape(1, -1), neighbor_embedding.reshape(1, -1))[0][0]
                                candidate_neighbors.append((neighbor_id, score))
                            # else: logging.warning(f"No embedding found for neighbor {neighbor_id}. Skipping.")

                    # Sort neighbors by score (higher similarity is better) - Bandit Exploitation
                    candidate_neighbors.sort(key=lambda x: x[1], reverse=True)
                    frontiers[seed_id] = candidate_neighbors
                    logging.debug(f"Seed {seed_id}: Found {len(candidate_neighbors)} new neighbors for node {current_node_id}.")

                # 2. Select Next Node from Frontier
                if not frontiers[seed_id]:
                    logging.info(f"Seed {seed_id}: No unvisited neighbors found from node {current_node_id}. Path might end here.")
                    # Mark as low collection for stopping check
                    collection_history[seed_id].append(0)
                else:
                    # Select the best scoring neighbor (simple greedy approach)
                    next_node_id, score = frontiers[seed_id].pop(0) # Get and remove best
                    logging.info(f"Seed {seed_id}: Moving from {current_node_id} to neighbor {next_node_id} (Score: {score:.4f})")
                    current_nodes[seed_id] = next_node_id
                    visited_nodes[seed_id].add(next_node_id)

                    # 3. Fetch Documents for the New Node
                    new_docs = self.get_topic_docs(next_node_id, max_docs=max_docs_per_topic)

                    # 4. Check Novelty and Update Buffer
                    if new_docs:
                        novel_documents, novel_embeddings = self._calculate_novelty(
                            new_docs,
                            buffer_embeddings[seed_id],
                            novelty_threshold,
                            max_buffer_check
                        )

                        if novel_documents:
                            logging.info(f"Seed {seed_id}, Node {next_node_id}: Found {len(novel_documents)} novel documents.")
                            document_buffers[seed_id].extend(novel_documents)
                            # Update buffer embeddings
                            if buffer_embeddings[seed_id] is None:
                                buffer_embeddings[seed_id] = np.array(novel_embeddings)
                            else:
                                buffer_embeddings[seed_id] = np.vstack([buffer_embeddings[seed_id], np.array(novel_embeddings)])
                            collection_history[seed_id].append(len(novel_documents))
                        else:
                            logging.info(f"Seed {seed_id}, Node {next_node_id}: Documents found but none were novel.")
                            collection_history[seed_id].append(0)
                    else:
                        logging.info(f"Seed {seed_id}, Node {next_node_id}: No documents found for this topic.")
                        collection_history[seed_id].append(0)

                # 5. Check Stopping Condition for this seed path
                # Calculate average collection rate over the history window
                if len(collection_history[seed_id]) == stop_patience:
                    avg_recent_collection = sum(collection_history[seed_id]) / stop_patience
                    if avg_recent_collection < min_new_docs_rate:
                         consecutive_low_collection[seed_id] += 1
                         logging.info(f"Seed {seed_id}: Low collection rate detected (Avg: {avg_recent_collection:.2f} < {min_new_docs_rate}). Patience: {consecutive_low_collection[seed_id]}/{stop_patience}")
                         if consecutive_low_collection[seed_id] >= stop_patience:
                             seeds_to_deactivate.add(seed_id)
                             logging.info(f"--- Stopping crawl for Seed Path {seed_id} due to low collection rate. ---")
                    else:
                         consecutive_low_collection[seed_id] = 0 # Reset patience if rate recovers


            # Update active seeds set
            active_seeds -= seeds_to_deactivate

            # Small delay to avoid overwhelming resources/APIs if applicable
            time.sleep(0.1)

        logging.info(f"\n--- Crawl Finished ---")
        total_docs = sum(len(docs) for docs in document_buffers.values())
        logging.info(f"Collected a total of {total_docs} novel documents across {len(seed_topics)} seed paths.")
        for seed, docs in document_buffers.items():
             logging.info(f"  Seed {seed}: Collected {len(docs)} documents.")

        return document_buffers

# --- Example Usage ---
if __name__ == "__main__":
    # Initialize the search object
    # Set load_local_model=False if you need to train a new model (will take time)
    # Ensure the path "old/bert_topic.pkl" exists if load_local_model=True
    try:
        signal_search_instance = SignalSearch(load_local_model=True, model_path="old/bert_topic.pkl")

        # --- Test get_topic_docs ---
        print("\n--- Testing get_topic_docs ---")
        example_topic_id = signal_search_instance.topic_df['Topic'].iloc[1] # Get the first actual topic ID (skip -1)
        print(f"Getting documents for Topic ID: {example_topic_id}")
        docs_for_topic = signal_search_instance.get_topic_docs(example_topic_id, max_docs=3)
        if docs_for_topic:
            print(f"Found {len(docs_for_topic)} documents (showing max 3):")
            for i, doc in enumerate(docs_for_topic):
                print(f"  Doc {i+1}: {doc[:100]}...") # Print snippet
        else:
            print("No documents found for this topic.")

         # --- Test get_topic_relationships ---
        print("\n--- Testing get_topic_relationships ---")
        print(f"Getting relationships for Topic ID: {example_topic_id}")
        relationships = signal_search_instance.get_topic_relationships(example_topic_id, debug=True)
        print(f"Related Topic IDs (filtered): {relationships}")


        # --- Run the Crawl ---
        print("\n--- Starting Crawl ---")
        user_query = "skin cream for wrinkles"
        collected_data = signal_search_instance.crawl(
            query=user_query,
            n_seed_topics=5,       # Start with 5 most relevant topics
            max_steps=15,          # Limit exploration depth/breadth per seed
            novelty_threshold=0.75, # Lower threshold -> more docs considered novel
            stop_patience=4,       # Stop if avg collection < 1 doc/step for 4 steps
            min_new_docs_rate=1,
            max_docs_per_topic=20, # Process max 20 docs from each visited node
            max_buffer_check=300   # Compare new docs against max 300 buffer docs for speed
        )

        print("\n--- Crawl Results ---")
        if collected_data:
            for seed_topic, documents in collected_data.items():
                print(f"Seed Topic {seed_topic}: Collected {len(documents)} novel documents.")
                # Optionally print some collected document snippets
                # if documents:
                #     print("  Examples:")
                #     for i, doc in enumerate(documents[:2]):
                #          print(f"    - {doc[:150]}...")

            print(collected_data)
        else:
            print("No documents were collected during the crawl.")

    except Exception as e:
        logging.exception(f"An error occurred during execution: {e}")