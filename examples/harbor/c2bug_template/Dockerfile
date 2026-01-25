# c2bug task environment
FROM {docker_image}

WORKDIR {workdir}

COPY bug.patch /tmp/c2bug_bug.patch

RUN mkdir -p /logs && \
    (git apply --verbose /tmp/c2bug_bug.patch \
        || git apply --verbose --reject /tmp/c2bug_bug.patch \
        || patch --batch --fuzz=5 -p1 -i /tmp/c2bug_bug.patch) \
    && if ! git diff --quiet; then \
         git add -A && \
         git -c user.email='c2bug@example.com' -c user.name='c2bug' \
         commit -m 'c2bug bug patch'; \
       fi \
    && rm -f /tmp/c2bug_bug.patch
