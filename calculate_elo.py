import logging
import os
from datetime import datetime
from decimal import Decimal
from typing import List

import boto3
from boto3.dynamodb.conditions import Attr, Key
from datasets import Dataset

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# Create a DynamoDB client
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')


def _create_arena_table():
    dynamodb.create_table(
        TableName='oaaic_chatbot_arena',
        KeySchema=[
            {
                'AttributeName': 'arena_battle_id',
                'KeyType': 'HASH'
            },
        ],
        AttributeDefinitions=[
            {
                'AttributeName': 'arena_battle_id',
                'AttributeType': 'S'
            },
            {
                'AttributeName': 'timestamp',
                'AttributeType': 'S'
            },
        ],
        ProvisionedThroughput={
            'ReadCapacityUnits': 5,
            'WriteCapacityUnits': 5
        },
        GlobalSecondaryIndexes=[
            {
                'IndexName': 'TimestampIndex',
                'KeySchema': [
                    {
                        'AttributeName': 'arena_battle_id',
                        'KeyType': 'HASH'
                    },
                    {
                        'AttributeName': 'timestamp',
                        'KeyType': 'RANGE'
                    },
                ],
                'Projection': {
                    'ProjectionType': 'ALL',
                },
                'ProvisionedThroughput': {
                    'ReadCapacityUnits': 5,
                    'WriteCapacityUnits': 5,
                }
            },
        ]
    )

def _create_elo_scores_table():
    dynamodb.create_table(
        TableName='elo_scores',
        KeySchema=[
            {
                'AttributeName': 'chatbot_name',
                'KeyType': 'HASH'  # Partition key
            },
        ],
        AttributeDefinitions=[
            {
                'AttributeName': 'chatbot_name',
                'AttributeType': 'S'
            },
        ],
        ProvisionedThroughput={
            'ReadCapacityUnits': 5,
            'WriteCapacityUnits': 5
        }
    )


def _create_elo_logs_table():
    dynamodb.create_table(
        TableName='elo_logs',
        KeySchema=[
            {
                'AttributeName': 'arena_battle_id',
                'KeyType': 'HASH'  # Partition key
            },
            {
                'AttributeName': 'battle_timestamp',
                'KeyType': 'RANGE'  # Sort key
            },
        ],
        AttributeDefinitions=[
            {
                'AttributeName': 'arena_battle_id',
                'AttributeType': 'S'
            },
            {
                'AttributeName': 'battle_timestamp',
                'AttributeType': 'S'
            },
            {
                'AttributeName': 'all',
                'AttributeType': 'S'
            }
        ],
        ProvisionedThroughput={
            'ReadCapacityUnits': 10,
            'WriteCapacityUnits': 10
        },
        GlobalSecondaryIndexes=[
            {
                'IndexName': 'AllTimestampIndex',
                'KeySchema': [
                    {
                        'AttributeName': 'all',
                        'KeyType': 'HASH'  # Partition key for the GSI
                    },
                    {
                        'AttributeName': 'battle_timestamp',
                        'KeyType': 'RANGE'  # Sort key for the GSI
                    }
                ],
                'Projection': {
                    'ProjectionType': 'ALL'
                },
                'ProvisionedThroughput': {
                    'ReadCapacityUnits': 10,
                    'WriteCapacityUnits': 10
                }
            },
        ]
    )


def get_unprocessed_battles(last_processed_timestamp):
    # Use boto3 to create a DynamoDB resource and reference the table
    table = dynamodb.Table('oaaic_chatbot_arena')

    # Use a query to retrieve unprocessed battles in temporal order
    response = table.scan(
        FilterExpression=Attr('timestamp').gt(last_processed_timestamp),
        # ScanIndexForward=True
    )

    return response['Items']


def calculate_elo(rating1, rating2, result, K=32):
    # Convert ratings to float
    rating1 = float(rating1)
    rating2 = float(rating2)

    # Calculate the expected outcomes
    expected_outcome1 = 1.0 / (1.0 + 10.0 ** ((rating2 - rating1) / 400.0))
    expected_outcome2 = 1.0 - expected_outcome1

    # Calculate the new Elo ratings
    new_rating1 = rating1 + K * (result - expected_outcome1)
    new_rating2 = rating2 + K * ((1.0 - result) - expected_outcome2)

    return Decimal(new_rating1).quantize(Decimal('0.00')), Decimal(new_rating2).quantize(Decimal('0.00'))


def get_last_processed_timestamp():
    table = dynamodb.Table('elo_logs')

    # Scan the table sorted by timestamp in descending order
    response = table.query(
        IndexName='AllTimestampIndex',
        KeyConditionExpression=Key('all').eq('ALL'),
        ScanIndexForward=False,
        Limit=1
    )

    # If there are no items in the table, return a default timestamp
    if not response['Items']:
        return '1970-01-01T00:00:00'

    # Otherwise, return the timestamp of the latest item
    return response['Items'][0]['battle_timestamp']


def log_elo_update(arena_battle_id, battle_timestamp, new_rating1, new_rating2):
    # Reference the elo_logs table
    table = dynamodb.Table('elo_logs')

    # Update the table
    table.put_item(
        Item={
            'arena_battle_id': arena_battle_id,
            'battle_timestamp': battle_timestamp,  # Use the timestamp of the battle
            'log_timestamp': datetime.now().isoformat(),  # Also store the timestamp of the log for completeness
            'new_rating1': new_rating1,
            'new_rating2': new_rating2,
            'all': 'ALL',
        }
    )


def get_elo_score(chatbot_name, elo_scores):
    if chatbot_name in elo_scores:
        return elo_scores[chatbot_name]

    table = dynamodb.Table('elo_scores')
    response = table.get_item(Key={'chatbot_name': chatbot_name})

    # If there is no item in the table, return a default score
    if 'Item' not in response:
        return 1500

    return response['Item']['elo_score']


def update_elo_score(chatbot_name, new_elo_score):
    table = dynamodb.Table('elo_scores')

    # This will create a new item if it doesn't exist
    table.put_item(
        Item={
            'chatbot_name': chatbot_name,
            'elo_score': Decimal(str(new_elo_score)),
        }
    )


def get_elo_scores():
    table = dynamodb.Table('elo_scores')

    response = table.scan()
    data = response['Items']

    return data


def _backfill_logs():
    table = dynamodb.Table('elo_logs')

    # Initialize the scan operation
    response = table.scan()

    for item in response['Items']:
        table.update_item(
            Key={
                'arena_battle_id': item['arena_battle_id'],
                'battle_timestamp': item['battle_timestamp']
            },
            UpdateExpression="SET #all = :value",
            ExpressionAttributeNames={
                '#all': 'all'
            },
            ExpressionAttributeValues={
                ':value': 'ALL'
            }
        )

def main():
    last_processed_timestamp = get_last_processed_timestamp()
    battles: List[dict] = get_unprocessed_battles(last_processed_timestamp)
    battles = sorted(battles, key=lambda x: x['timestamp'])
    elo_scores = {}

    for battle in battles:
        print(repr(battle))
        if battle['label'] in {-1, 0, 1, 2}:
            outcome = battle['label']
            for chatbot_name in [battle['choice1_name'], battle['choice2_name']]:
                if chatbot_name not in elo_scores:
                    elo_scores[chatbot_name] = get_elo_score(chatbot_name, elo_scores)
            # 1: This means that the first player (or team) won the match.
            # 0.5: This means that the match ended in a draw.
            # 0: This means that the first player (or team) lost the match.
            if outcome == 0 or outcome == -1:
                elo_result = 0.5
            elif outcome == 1:
                elo_result = 1
            else:
                elo_result = 0

            new_rating1, new_rating2 = calculate_elo(elo_scores[battle['choice1_name']], elo_scores[battle['choice2_name']], elo_result)
            logging.info(f"{battle['choice1_name']}: {elo_scores[battle['choice1_name']]} -> {new_rating1} | {battle['choice2_name']}: {elo_scores[battle['choice2_name']]} -> {new_rating2}")
            elo_scores[battle['choice1_name']] = new_rating1
            elo_scores[battle['choice2_name']] = new_rating2
            log_elo_update(battle['arena_battle_id'], battle['timestamp'], new_rating1, new_rating2)
            update_elo_score(battle['choice1_name'], new_rating1)
            update_elo_score(battle['choice2_name'], new_rating2)
            elo_scores[battle['choice1_name']] = new_rating1
            elo_scores[battle['choice2_name']] = new_rating2

    elo_scores = get_elo_scores()
    for i, j in enumerate(elo_scores):
        j["elo_score"] = float(j["elo_score"])
        elo_scores[i] = j
    print(elo_scores)

    if battles:
        # Convert the data into a format suitable for Hugging Face Dataset
        elo_dataset = Dataset.from_list(elo_scores)
        elo_dataset.push_to_hub("openaccess-ai-collective/chatbot-arena-elo-scores", private=False)


if __name__ == "__main__":
    main()
