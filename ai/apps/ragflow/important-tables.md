# Important tables

These tables (`user`, `tenant`, `user_tenant`, `knowledgebase`, `file`, `document`, `file2document`, `llm`, `tenant_llm`, `api_token`) is very critical

## `user`
It include all user information

## `tenant`
It include all tenant information.
When create new user, it will create tenant which id is same as user.

## `user_tenant`
Maintain user tenant relationship.
- When create new user, it will create user_tenant which role is owner.
- When invite someone, it will create user_tent which role is **invite**.
- When invitee is accepted, the role became **normal**.

## `knowledgebase`
The knowledgebase and dataset are same concept.

## `file`
The file contain file and folder information and hierarchy.
The parent_id is the parent folder in where the file is placed.

## `file2document`
The file2document save relationship between file and document.

## `document`
The document save meta info about file.
Can use `kb_id` and `location` to find file in **Minio**.

## `llm`
Save the supported model provider

## `tenant_llm`
Save the customized LLM

## `api_token`
Include the token used by **ragflow-sdk** and API