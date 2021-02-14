import time
from threading import Thread, Event
from queue import Queue
from threading import Lock


def setup_mock_executor(gpu_list, jobs_per_gpu):

    return MockExecutor(gpu_list, jobs_per_gpu)


def check_jobs_done_mock(jobs, executor):

    while True:

        num_finished = sum(executor.check_job_done(job) for job in jobs)

        if num_finished == len(jobs):
            break

        time.sleep(5)


class MockExecutor:

    def __init__(self, gpu_list, jobs_per_gpu):

        self.gpu_list = gpu_list
        self.jobs_per_gpu = jobs_per_gpu

        # fc_queue is for jobs to run
        self.fc_queue = Queue()
        # gpu_queue is for available gpus
        self.gpu_queue = Queue()

        # done dict indicates which jobs are done
        self.done_dict = dict()
        self.done_dict_lock = Lock()

        # running list keeps track of running jobs
        self.running_list = list()
        self.running_list_lock = Lock()

        # each job gets an index
        self.running_job_idx = 0

        # enqueue available gpus and start worker threads
        self.enqueue_gpus_()
        self.worker_run_thread, self.worker_release_thread = None, None
        self.worker_release_thread_flag = Event()
        self.run_threads_()

    def submit(self, run_fc, *args, **kwargs):

        job_idx = self.running_job_idx
        self.running_job_idx += 1

        self.done_dict_lock.acquire()
        self.done_dict[job_idx] = False
        self.done_dict_lock.release()

        self.fc_queue.put((run_fc, job_idx, args, kwargs))

        return job_idx

    def check_job_done(self, job_idx):

        self.done_dict_lock.acquire()
        done = self.done_dict[job_idx]
        self.done_dict_lock.release()

        return done

    def stop(self):

        self.fc_queue.put(None)
        self.worker_release_thread_flag.set()

        self.worker_run_thread.join()
        self.worker_release_thread.join()

    def enqueue_gpus_(self):

        for gpu in self.gpu_list:
            for _ in range(self.jobs_per_gpu):
                self.gpu_queue.put(gpu)

    def run_threads_(self):

        self.worker_run_thread = Thread(target=self.worker_run_)
        self.worker_run_thread.start()

        self.worker_release_thread = Thread(target=self.worker_release_)
        self.worker_release_thread.start()

    def worker_run_(self):

        while True:

            item = self.fc_queue.get()

            if item is None:
                break
            else:
                run_fc, job_idx, args, kwargs = item

            gpu_idx = self.gpu_queue.get()

            kwargs["gpu"] = gpu_idx
            process = run_fc(*args, **kwargs)

            self.running_list_lock.acquire()
            self.running_list.append((job_idx, gpu_idx, process))
            self.running_list_lock.release()

    def worker_release_(self):

        while True:

            self.running_list_lock.acquire()

            if self.worker_release_thread_flag.is_set() and len(self.running_list) == 0:
                self.running_list_lock.release()
                break

            to_delete = []

            for idx, item in enumerate(self.running_list):

                job_idx, gpu_idx, process = item

                if process.poll() is not None:

                    self.done_dict_lock.acquire()
                    self.done_dict[job_idx] = True
                    self.done_dict_lock.release()

                    self.gpu_queue.put(gpu_idx)

                    to_delete.append(idx)

            for idx in reversed(to_delete):

                del self.running_list[idx]

            self.running_list_lock.release()

            # there's no queue so better sleep
            time.sleep(5)
