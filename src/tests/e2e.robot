*** Settings ***
Library    Process
Library    RequestsLibrary
Library    Collections
Suite Setup    Start Server
Suite Teardown    Stop Server

*** Variables ***
${BASE_URL}    http://localhost:8080

*** Test Cases ***
Get main images array
    [Documentation]    Fetches the main array of images from free-images.com and validates the response.
    ${response}=    GET    ${BASE_URL}/images/main
    Should Be Equal As Strings    ${response.status_code}    200
    ${json_response}=    Evaluate    json.loads($response.content)    json
    Should Be True    isinstance($json_response, list)
    Should Be True    len($json_response) > 0
    # Validate that each item in the array is a list of image URLs
    FOR    ${image_category}    IN    @{json_response}
        Should Be True    isinstance($image_category, list)
        FOR    ${image_url}    IN    @{image_category}
            Should Be True    isinstance($image_url, str)
            Should Match Regexp    ${image_url}    ^^\/.*\/.*\/.*\\.(jpg|jpeg|png|gif|webp)
        END
    END

Get subject of an image
    [Documentation]    Gets the subject of a specific image and validates the response.
    ${params}=    Create Dictionary    image_url=/md/7461/violet_viola_purple_plant.jpg
    ${response}=    GET    ${BASE_URL}/images/subject    params=${params}
    Should Be Equal As Strings    ${response.status_code}    200
    ${json_response}=    Evaluate    json.loads($response.content)    json
    Dictionary Should Contain Key    ${json_response}    subject
    Should Be True    isinstance($json_response['subject'], str)
    Should Be True    len($json_response['subject']) > 0

Get subject of an image with missing parameter
    [Documentation]    Tests error handling when image_url parameter is missing.
    ${response}=    GET    ${BASE_URL}/images/subject    expected_status=422
    Should Be Equal As Strings    ${response.status_code}    422
    ${json_response}=    Evaluate    json.loads($response.content)    json
    Dictionary Should Contain Key    ${json_response}    detail

Get images from Google with default parameters
    [Documentation]    Gets images from Google Images with default max_images and validates the response.
    ${params}=    Create Dictionary    subject=cats
    ${response}=    GET    ${BASE_URL}/images/google    params=${params}
    Should Be Equal As Strings    ${response.status_code}    200
    ${json_response}=    Evaluate    json.loads($response.content)    json
    Should Be True    isinstance($json_response, list)
    Should Be True    len($json_response) <= 10
    # Validate that each item is an image URL
    FOR    ${image_url}    IN    @{json_response}
        Should Be True    isinstance($image_url, str)
        Should Match Regexp    ${image_url}    ^https?://.*
    END

Get images from Google with custom max_images
    [Documentation]    Gets images from Google Images with custom max_images parameter and validates the response.
    ${params}=    Create Dictionary    subject=dogs    max_images=5
    ${response}=    GET    ${BASE_URL}/images/google    params=${params}
    Should Be Equal As Strings    ${response.status_code}    200
    ${json_response}=    Evaluate    json.loads($response.content)    json
    Should Be True    isinstance($json_response, list)
    Should Be True    len($json_response) <= 5
    # Validate that each item is an image URL
    FOR    ${image_url}    IN    @{json_response}
        Should Be True    isinstance($image_url, str)
        Should Match Regexp    ${image_url}    ^https?://.*
    END

Get images from Google with missing subject parameter
    [Documentation]    Tests error handling when subject parameter is missing.
    ${response}=    GET    ${BASE_URL}/images/google    expected_status=422
    Should Be Equal As Strings    ${response.status_code}    422
    ${json_response}=    Evaluate    json.loads($response.content)    json
    Dictionary Should Contain Key    ${json_response}    detail

Test root endpoint redirect
    [Documentation]    Tests that the root endpoint redirects to /docs.
    ${response}=    GET    ${BASE_URL}/    allow_redirects=${False}
    Should Be True    ${response.status_code} in [301, 302, 307, 308]
    Dictionary Should Contain Key    ${response.headers}    location
    Should Contain    ${response.headers['location']}    /docs

*** Keywords ***
Start Server
    ${result}=    Start Process    uvicorn    src.main:app    --host    0.0.0.0    --port    8080    alias=fastapi_server
    Log    Server started with PID: ${result.pid}
    Sleep    5s

Stop Server
    Terminate Process    fastapi_server
