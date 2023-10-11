import logging

import weaviate  # type: ignore[import]
import json
from constant import data_path
import logging
from flask import after_this_request, jsonify
import constant
import os
import openai
from pathlib import Path
from tqdm.notebook import tqdm
from wasabi import msg  # type: ignore[import]
from flask import Flask, request
from dotenv import load_dotenv
from utils import parse_text, parse_embeddings, convert_all_jsons_to_dataframe, weaviate_client
from abc import abstractmethod
from sentence_transformers import SentenceTransformer

load_dotenv()

model = SentenceTransformer(constant.MODEL_NAME)


class EmbeddingBaseService:
    def __init__(self):
        super().__init__(__file__)

    def _initialize(self, file_name):
        # Initialize base variables
        self.SCHEMA_NAME = constant.SCHEMA_NAME
        self.model = constant.MODEL_NAME
        self.data_path = constant.data_path
        self.app = Flask(self.name)
        self._initialize_class_variables()

    @staticmethod
    def handle_service_error(fail_code=None):
        def func(error, _fail_code=fail_code):
            logging.ERROR(repr(error), exc_info=True)
            ret = {
                'status': 'failure',
                'response': {"error_message": repr(error)}
            }
            response = jsonify(ret)
            response.status_code = fail_code if _fail_code is not None else 500
            return response

        return func

    def _initialize_class_variables(self):
        self.ip = '0.0.0.0'
        self.supported_methods = ['GET', 'POST']
        self.port = None
        self.route = None
        self.service_name = None
        # self.app.config.from_object(self.name)
        self.protocol = None
        self.do_unload = False
        self.collect_garbage = False
        self.model_access_timer = {}
        self.models = {}
        self.flag_func_map = None

    @abstractmethod
    def initialize_service(self):
        pass

    def authenticate_and_parse_request(self):
        input_data = self._parse_request()
        status_code = 200

        logging.info("Request has been parsed")
        logging.info('Current request being served from: {}'.format(request.path))
        logging.debug(input_data)

        if status_code == 200:
            return input_data
        else:
            raise Exception('Invalid request')

    @staticmethod
    def _parse_request():
        try:
            input_text = request.get_data(as_text=True)

        except:
            input_text = None
            logging.exception("Cannot parse input request")

        return input_text

    def EmbeddedBatch(self):

        df = convert_all_jsons_to_dataframe(self.data_path)
        df["Index"] = df.index
        df["Embedding Text"] = df.apply(parse_text, axis=1)
        df["Embeddings"] = parse_embeddings(df["Embedding Text"].tolist(), model)
        client = weaviate.Client(
            embedded_options=weaviate.embedded.EmbeddedOptions(),
            additional_headers={
                # 'X-OpenAI-Api-Key': 'YOUR-OPENAI-API-KEY'  # Replace w/ your OPENAI API key,
                # "X-Huggingface-Api-Key": "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"
            }
        )

        weaviate_client().schema.create_class({
            'class': self.SCHEMA_NAME,
            "description": "Patent Data Information",
            # 'vectorizer': 'text2vec-openai',
            # 'vectorizer': 'text2vec-huggingface'
            'vectorizer': 'none'
        })
        # for object_ in df.to_dict("r")[:5]:
        #     client.data_object.create({
        #         'name': 'Chardonnay',
        #         'review': 'Goes well with fish!',
        #     }, SCHEMA_NAME)

        weaviate_client().batch.configure(batch_size=100)  # Configure batch
        with client.batch as batch:
            # Batch import all entries
            for i, d in tqdm(enumerate(df.to_dict("records")), total=len(df)):
                properties = {
                    "index": d["Index"],
                    "patent_number": d["patent_number"],
                    "family_id": d["family_id"],
                    "inventors": d["inventors"],
                    "assignees": d["assignees"],
                    "publication_id": d["publication_id"],
                    "publication_date": d["publication_date"],
                    "titles": d["titles"],
                    "abstracts": d["abstracts"],
                    "claims": d["claims"],
                    "descriptions": d["descriptions"],
                    "national_classes": d["national_classes"],
                    "ipc_classes": d["ipc_classes"],
                    "locarno_classes": d["locarno_classes"],
                    "ipcr_classes": d["ipcr_classes"],
                    "ecla_classes": d["ecla_classes"],
                    "cpc_classes": d["cpc_classes"],
                    "f_term_classes": d["f_term_classes"],
                    "legal_status": d["legal_status"],
                    "priority_date": d["priority_date"],
                    "application_date": d["application_date"],
                    "family_members": d["family_members"],
                    "full_text": d["Embedding Text"]
                }

                batch.add_data_object(properties, self.SCHEMA_NAME, vector=d["Embeddings"])

        return batch
