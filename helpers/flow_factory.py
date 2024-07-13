from flows.AbstractFlow import AbstractFlow
from flows.community_detection.CommunityDetectionFlow import CommunityDetectionFlow
from flows.phewas_analysis.PhewasAnalysisFlow import PhewasAnalysisFlow
from helpers.consts import *


def get_flow(type: str) -> AbstractFlow:
    if type == 'community_detection':
        return CommunityDetectionFlow(type)
    elif type == 'phewas_analysis':
        return PhewasAnalysisFlow(type)
    else:
        raise Exception(f'Flow Type {type} does not exist')
