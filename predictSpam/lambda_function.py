import os
import io
import boto3
import json
import csv
import urllib.parse
import email
import re
from sms_spam_classifier_utilities import one_hot_encode
from sms_spam_classifier_utilities import vectorize_sequences


vocabulary_length = 9013

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime = boto3.client('runtime.sagemaker')


def lambda_handler(event, context):
    s3 = boto3.client('s3')
    
    # Get s3 object contents based on bucket name and object key; in bytes and convert to string
    # key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    data = s3.get_object(Bucket="emails-as3", Key=event['Records'][0]['s3']['object']['key'])
    contents = data['Body'].read().decode("utf-8")
    
    # Given the s3 object content is the ses email, get the message content and attachment using email package
    msg = email.message_from_string(contents)
    date = msg['Date']
    subject = msg['Subject']
    
    # get the content of this email
    attachment = msg.get_payload()[0]
    original = attachment.get_payload()
    formatted = original.replace("\n"," ")
    formatted = formatted.replace("\r"," ")
    body = ' '.join(formatted.split())
    
    # get the email of the sender
    fromAddress = msg['from']
    regex = "\\<(.*?)\\>"
    fromAddress = re.findall(regex, fromAddress)[0]
    
    # print("body: ",body)
    # print("from: ",fromAddress)
    # print("date: ",date)
    # print("subject: ", subject)

    # predict whether this email is spam
    emails = [body]
    one_hot_emails = one_hot_encode(emails, vocabulary_length)
    encoded_emails = vectorize_sequences(one_hot_emails, vocabulary_length)
    payload = json.dumps(encoded_emails.tolist())
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                      ContentType='application/json',
                                      Body=payload)
    result = json.loads(response['Body'].read().decode())
    pred_label = result['predicted_label'][0][0]
    pred_probability =result['predicted_probability'][0][0] 
    if pred_label==0:
        pred_class="Ham"
    else:
        pred_class="Spam"
    # print("the class is {} with a probability of {:2.2f}%".format(pred_class,pred_probability*100))

    # send response to sender
    p1 = "We received your email sent at {} with the subject {}.".format(date,subject)
    if len(body)<=240:
        p2 = "Here is a 240 character sample of the email body: "+body
    else:
        p2 = "Here is a 240 character sample of the email body: "+body[:240]
    p3 = "The email was categorized as {} with a {:2.2f}% confidence".format(pred_class,pred_probability*100)
    body_text = "\n".join([p1,p2,p3])
    body_html =  '<p>'+p1+'</p> '+'<p>'+p2+'</p> '+'<p>'+p3+'</p> '
    
    source_email = 'xiaoyutian@cloudcomputing6998-as3.com'
    client = boto3.client('ses')
    response = client.send_email(
    Destination={
        'ToAddresses': [
            fromAddress
        ],
    },
    Message={
        'Body': {
            'Html': {
                'Charset': 'UTF-8',
                'Data': body_html,
            },
            'Text': {
                'Charset': 'UTF-8',
                'Data': body_text,
            },
        },
        'Subject': {
            'Charset': 'UTF-8',
            'Data': 'Spam Detection',
        },
    },
    Source=source_email,
    )
    message_id = response['MessageId']

    return {
        "prediction label":pred_label,
        "message id":message_id
    }