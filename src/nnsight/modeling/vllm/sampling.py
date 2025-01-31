import copy
from typing import Dict, List, Optional, Tuple

from nnsight.intervention.graph import InterventionGraph
import torch

from vllm.model_executor.sampling_metadata import (
    SamplingMetadata,
    SamplingMetadataCache,
    _prepare_seq_groups,
)
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceGroupMetadata
from vllm.utils import async_tensor_h2d


class NNsightSamplingParams(SamplingParams):

    intervention_graph: Optional[InterventionGraph] = None
    nns_batch_groups: Optional[List[Tuple[int, int]]] = None
    invoker_group: Optional[int] = None
    is_default_param: bool = True

    def clone(self) -> "SamplingParams":
        """Deep copy excluding LogitsProcessor objects.

        LogitsProcessor objects are excluded because they may contain an
        arbitrary, nontrivial amount of data.
        See https://github.com/vllm-project/vllm/issues/3087
        """

        memo = {}

        if self.logits_processors is not None:
            for lp in self.logits_processors:
                memo[id(lp)] = lp

        if self.intervention_graph is not None:
            memo[id(self.intervention_graph)] = self.intervention_graph

        return copy.deepcopy(self, memo=memo)


class NNsightSamplingMetadata(SamplingMetadata):

    intervention_graph: Optional[InterventionGraph] = None
    nns_batch_groups: Optional[List[Tuple[int, int]]] = None
    batch_groups: Optional[List[Tuple[int, int]]] = None

    def __init__(
        self,
        *args,
        intervention_graph: InterventionGraph = None,
        nns_batch_groups: List[Tuple[int, int]] = None,
        batch_groups: Dict[int, Tuple[int, int]] = None,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        self.intervention_graph = intervention_graph
        self.nns_batch_groups = nns_batch_groups
        self.batch_groups = batch_groups

    @staticmethod
    def prepare(
        seq_group_metadata_list: List[SequenceGroupMetadata],
        seq_lens: List[int],
        query_lens: List[int],
        device: str,
        pin_memory: bool,
        generators: Optional[Dict[str, torch.Generator]] = None,
        cache: Optional[SamplingMetadataCache] = None,
    ) -> "SamplingMetadata":
        (
            seq_groups,
            selected_token_indices,
            categorized_sample_indices,
            num_prompts,
        ) = _prepare_seq_groups(
            seq_group_metadata_list, seq_lens, query_lens, device, generators, cache
        )
        selected_token_indices = async_tensor_h2d(
            selected_token_indices,
            dtype=torch.long,
            target_device=device,
            pin_memory=pin_memory,
        )
        categorized_sample_indices = {
            t: async_tensor_h2d(
                seq_ids,
                dtype=torch.int,
                target_device=device,
                pin_memory=pin_memory,
            )
            for t, seq_ids in categorized_sample_indices.items()
        }
        
        
        ### NNSIGHT ###########################################

        intervention_graphs = []
        nns_batch_groups = []
        batch_groups = []
        batch_groups_offset = 0

        for idx, seq_group in enumerate(seq_group_metadata_list):

            if isinstance(seq_group.sampling_params, NNsightSamplingParams):

                seq_group_intervention_graph = (
                    seq_group.sampling_params.intervention_graph
                )

                seq_group_nns_batch_groups = seq_group.sampling_params.nns_batch_groups
                
                if isinstance(seq_group_intervention_graph, InterventionGraph):
                    
                    if seq_group_intervention_graph not in intervention_graphs:
                    
                        intervention_graphs.append(seq_group_intervention_graph)

                        nns_batch_groups.append(seq_group_nns_batch_groups)

                        batch_groups_offset = len(batch_groups)

                    seq_group_batch_group = (
                        seq_group.sampling_params.invoker_group + batch_groups_offset
                    )

                    batch_size = query_lens[idx]
            
                    if seq_group_batch_group >= len(batch_groups):
                        batch_start = sum(batch_groups[-1]) if len(batch_groups) > 0 else 0
                        batch_groups.append((batch_start, batch_size))
                    else:
                        batch_start, seq_group_batch_size = batch_groups[
                            seq_group_batch_group
                        ]
                        batch_size += seq_group_batch_size

                        batch_groups[seq_group_batch_group] = (batch_start, batch_size)
                    
        n_graphs = len(intervention_graphs)
        
        if n_graphs== 0:
            intervention_graph = None
            nns_batch_groups = None
        elif n_graphs == 1:
            intervention_graph =intervention_graphs[0]
            nns_batch_groups = nns_batch_groups[0]

        """ else:
            intervention_graph = MultiGraph(intervention_graphs.values())
            
            InterventionProtocol.shift(intervention_graph) """

        ###########################################

        sampling_metadata = NNsightSamplingMetadata(
            seq_groups=seq_groups,
            selected_token_indices=selected_token_indices,
            categorized_sample_indices=categorized_sample_indices,
            num_prompts=num_prompts,
            intervention_graph=intervention_graph,
            nns_batch_groups = nns_batch_groups,
            batch_groups=batch_groups,
        )

        return sampling_metadata
