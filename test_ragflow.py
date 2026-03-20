#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
The example is about CRUD operations (Create, Read, Update, Delete) on a dataset.
"""

from ragflow_sdk import RAGFlow
import sys
import os

HOST_ADDRESS = os.getenv("RAGFLOW_SERVER_HOST")
API_KEY = os.getenv("RAGFLOW_API_KEY")

def main():
    # create a ragflow instance
    ragflow_instance = RAGFlow(api_key=API_KEY, base_url=HOST_ADDRESS)

    # crate a dataset instance
    dataset_instance = ragflow_instance.create_dataset(name="dataset_instance")
    print(dataset_instance)

    # update the dataset instance
    updated_message = {"name":"updated_dataset"}
    updated_dataset = dataset_instance.update(updated_message)

    # get the dataset (list datasets)
    print(updated_dataset)

    # delete the dataset (delete datasets)
    to_be_deleted_datasets = [dataset_instance.id]
    ragflow_instance.delete_datasets(ids=to_be_deleted_datasets)

    print([d.name for d in ragflow_instance.list_datasets()])  # list datasets after deletion, should be empty
    print([c.name for c in ragflow_instance.list_chats()])  # list chats after deletion, should be empty
    print("test done")

    #chat = ragflow_instance.list_chats(name="replace with chat name")[0]
    #if len(chat.list_sessions(name="replace with your session name")) == 0 or not (session := chat.list_sessions(name="replace with your session name")[0]):
    #    session = chat.create_session(name="replace with your session name")
    #resp = session.ask(question="Replace with your question", stream=False)
    #for r in resp:
    #    print(r)

def upload_to_ragflow(path: str, dataset_name: str = "TEST UPLOAD DATASET", alias: str = "temp.pdf"):
    ragflow_instance = RAGFlow(api_key=API_KEY, base_url=HOST_ADDRESS)
    for d in ragflow_instance.list_datasets(name=dataset_name):
        docs = []
        if path.endswith(".pdf"):
            with open(alias, "rb") as file:
                display_name = path
                blob = file.read()
            docs.append({"display_name": display_name, "blob": blob})
        else:
            for f in os.listdir(path):
                if f.endswith(".pdf"):
                    with open(os.path.join(path, f), "rb") as file:
                        display_name = f
                        blob = file.read()
                    docs.append({"display_name": display_name, "blob": blob})
        d.upload_documents(docs)

def exist_doc_in_ragflow(dataset_name: str, doc_name: str):
    ragflow_instance = RAGFlow(api_key=API_KEY, base_url=HOST_ADDRESS)
    for d in ragflow_instance.list_datasets(name=dataset_name):
        try:
            if len(d.list_documents(name=doc_name)) > 0:
                return True
        except Exception:
            continue
    return False

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(str(e))
        sys.exit(-1)


