"""test tasks"""
from study.demo import completion_task


def test_completeion_task():
    """test `completion_task`"""
    query = 'What is the answer to life the universe and everything?'
    ans = completion_task(query)
    print(ans)
