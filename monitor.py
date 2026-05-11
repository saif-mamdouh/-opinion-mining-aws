import boto3, time, psutil, datetime

cloudwatch = boto3.client('cloudwatch', region_name='eu-north-1')

def push_metrics():
    cloudwatch.put_metric_data(
        Namespace='OpinionMining',
        MetricData=[
            {
                'MetricName': 'CPUUsage',
                'Value': psutil.cpu_percent(),
                'Unit': 'Percent',
                'Timestamp': datetime.datetime.utcnow()
            },
            {
                'MetricName': 'MemoryUsage',
                'Value': psutil.virtual_memory().percent,
                'Unit': 'Percent',
                'Timestamp': datetime.datetime.utcnow()
            }
        ]
    )
    print(f"✅ Metrics pushed at {datetime.datetime.utcnow()}")

while True:
    push_metrics()
    time.sleep(60)
