import logging
import os
import weaviate  # type: ignore[import]
import json
import constant
from wasabi import msg  # type: ignore[import]
from sentence_transformers import SentenceTransformer
from embedding import EmbeddingBaseService
from dotenv import load_dotenv
import constant
from utils import weaviate_client,parse_text, parse_embeddings

load_dotenv()

# Request Count
request_count = 0
cache_count = 0

# Configuration
data_fields = ['patent_number', 'publication_id', 'family_id', 'publication_date',
               'titles', 'abstracts', 'claims', 'descriptions', 'inventors',
               'assignees', 'ipc_classes', 'locarno_classes', 'ipcr_classes',
               'national_classes', 'ecla_classes', 'cpc_classes', 'f_term_classes',
               'legal_status', 'priority_date', 'application_date', 'family_members',
               'Index', 'full_text']

model = SentenceTransformer(constant.MODEL_NAME)


class SemanticSearchService(EmbeddingBaseService):

    def __init__(self):
        super().__init__(__file__)
        self.similarity = None
        self.service_name = None
        self.initialize_semantic_search_service()


    def initialize_service(self):
        self.service_name = 'Semantic_search_service'
        self.similarity = None

    def initialize_semantic_search_service(self):
        self.initialize_service()


    def get_semantic_response(self):
        try:
            input_query = self.authenticate_and_parse_request()
        except logging.exception():
            return self.error_msg

        response = (
            self.EmbeddedBatch().query
            .get(constant.SCHEMA_NAME,
                 ['patent_number', 'publication_id', 'family_id', 'publication_date', 'titles', 'abstracts', 'claims',
                  'descriptions', 'inventors','ipc_classes', 'locarno_classes', 'ipcr_classes','national_classes', 'ecla_classes', 'cpc_classes', 'f_term_classes',
                  'legal_status', 'priority_date', 'application_date', 'family_members',
                  'index', 'full_text'])
            # .with_additional(['certainty', "distance", "vector"])
            .with_near_vector({
                'vector': parse_embeddings(input_query, model)
            })
            .with_limit(1)
            .do()
        )

        return json.dumps(response, indent=4)



Semantic_search_service = SemanticSearchService()
app = Semantic_search_service.app


@Semantic_search_service.app.route(Semantic_search_service.route, methods=Semantic_search_service.supported_methods)
def handle_route1():
    return Semantic_search_service.get_semantic_response()




if __name__ == '__main__':
    Semantic_search_service.run()

    """

    http://localhost:8102/Semantic_search
    payload:
    {
        "sourceTexts":["Hello world"],
        "targetTexts":["Hi How are you"]
    }

    http://localhost:8102/nlp/similarity/get_embeddings
    payload:
    {
    "texts":["hot"]
    }

    """
