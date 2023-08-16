from typing import List
from google.cloud import batch_v1

PROJECT_ID = "ao3-disco"
REGION = "us-central1" # Every othe region has long wait times.

def create_container_job(job_name: str, train_args: List[str]=[]) -> batch_v1.Job:
    client = batch_v1.BatchServiceClient()

    # Define the Docker container that will be executed.
    runnable = batch_v1.Runnable()
    runnable.container = batch_v1.Runnable.Container()
    runnable.container.image_uri = "us-west1-docker.pkg.dev/ao3-disco/ao3-disco-docker/ao3-disco-ai"
    runnable.container.commands = ["train"] + train_args
    runnable.container.volumes = [
        "/mnt/disks/ao3-disco-ai/data:/app/data",
        "/mnt/disks/ao3-disco-ai/lightning_logs:/app/lightning_logs"
    ]

    # Jobs can be divided into tasks. In this case, we have only one task.
    task = batch_v1.TaskSpec()
    task.runnables = [runnable]

    # We can specify what resources are requested by each task.
    resources = batch_v1.ComputeResource()
    resources.cpu_milli = 2000  # in milliseconds per cpu-second. This means the task requires 2 whole CPUs.
    resources.memory_mib = 16000  # ~16GB
    task.compute_resource = resources

    # Mount the Google Cloud Storage bucket to the container.
    task.volumes = [
        {
            "gcs": {
                "remote_path": "ao3-disco-ai/data"
            },
            "mount_path": "/mnt/disks/ao3-disco-ai/data"
        },
        {
            "gcs": {
                "remote_path": "ao3-disco-ai/lightning_logs"
            },
            "mount_path": "/mnt/disks/ao3-disco-ai/lightning_logs"
        }
    ]

    task.max_retry_count = 0
    task.max_run_duration = "604800s" # 1 day

    # Tasks are grouped inside a job using TaskGroups.
    # Currently, it's possible to have only one task group.
    group = batch_v1.TaskGroup()
    group.task_count = 1
    group.task_spec = task

    # Policies are used to define on what kind of virtual machines the tasks will run on.
    # In this case, we tell the system to use "e2-standard-4" machine type.
    # Read more about machine types here: https://cloud.google.com/compute/docs/machine-types
    policy = batch_v1.AllocationPolicy.InstancePolicy()
    policy.machine_type = "c2-standard-4"
    policy.provisioning_model = batch_v1.AllocationPolicy.ProvisioningModel.SPOT
    instances = batch_v1.AllocationPolicy.InstancePolicyOrTemplate()
    instances.policy = policy
    allocation_policy = batch_v1.AllocationPolicy()
    allocation_policy.instances = [instances]

    job = batch_v1.Job()
    job.task_groups = [group]
    job.allocation_policy = allocation_policy

    # We use Cloud Logging as it's an out of the box available option
    job.logs_policy = batch_v1.LogsPolicy()
    job.logs_policy.destination = batch_v1.LogsPolicy.Destination.CLOUD_LOGGING

    create_request = batch_v1.CreateJobRequest()
    create_request.job = job
    create_request.job_id = job_name
    # The job's parent is the region in which the job will run
    create_request.parent = f"projects/{PROJECT_ID}/locations/{REGION}"

    return client.create_job(create_request)

if __name__ == "__main__":
    print(create_container_job(
        job_name="no-sim-loss", 
        train_args=["--similarity-loss-scale", "0.0"]
    ))
