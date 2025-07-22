*** Settings ***
Library    Process
Library    RequestsLibrary
Library    Collections
Suite Setup    Start Server
Suite Teardown    Stop Server

*** Variables ***
${BASE_URL}    http://localhost:8080

*** Test Cases ***
Get a random video
    [Documentation]    Fetches a random video and validates the response.
    ${body}=    Create Dictionary    query=get me a random video
    ${headers}=    Create Dictionary    Content-Type=application/json
    ${response}=    POST    ${BASE_URL}/video    json=${body}    headers=${headers}
    Should Be Equal As Strings    ${response.status_code}    200
    ${json_response}=    Evaluate    json.loads($response.content)    json
    Dictionary Should Contain Key    ${json_response}    url
    Dictionary Should Contain Key    ${json_response}    title
    Dictionary Should Contain Key    ${json_response}    uploader
    Dictionary Should Contain Key    ${json_response}    duration

Get a video from a specific channel
    [Documentation]    Fetches a video from a specific channel and validates the response.
    ${body}=    Create Dictionary    query=get me a video from the kid channel
    ${headers}=    Create Dictionary    Content-Type=application/json
    ${response}=    POST    ${BASE_URL}/video    json=${body}    headers=${headers}
    Should Be Equal As Strings    ${response.status_code}    200
    ${json_response}=    Evaluate    json.loads($response.content)    json
    Dictionary Should Contain Key    ${json_response}    url
    Dictionary Should Contain Key    ${json_response}    title
    Dictionary Should Contain Key    ${json_response}    uploader
    Dictionary Should Contain Key    ${json_response}    duration

*** Keywords ***
Start Server
    ${result}=    Start Process    uvicorn    src.main:app    --host    0.0.0.0    --port    8080    alias=fastapi_server
    Log    Server started with PID: ${result.pid}
    Sleep    5s

Stop Server
    Terminate Process    fastapi_server
