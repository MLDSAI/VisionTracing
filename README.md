# VisionTracing

VisionTracing is a web application for running keypoint detection, tracking, and subject re-identification on video files.

## Deployment

### Docker

(Requires Docker to be installed.)

Run locally or in the cloud with:

```
docker-compose up
```

### Heroku

(Requires a Heroku account and the Heroku CLI to be installed.)

Pick a suffix to differentiate this deployment from the one at visiontracing.herokuapp.com:

```
$ heroku apps:create visiontracing-<suffix>
```

Make sure that the Heroku remote was created:

```
$ git remote
heroku
origin
```

Configure the Heroku app stack to use Docker:

```
heroku stack:set container
```

Provision Redis:

```
$ heroku addons:create heroku-redis:hobby-dev
```

Provision Postgres:

```
$ heroku addons:create heroku-postgresql:hobby-dev
```

Provision at least one worker:

```
$ heroku ps:scale worker=1
```

Push to Heroku to deploy:

```
git push heroku master
```

Once deployment is complete, monitor the application logs:

```
heroku logs --tail
```

## Usage

The following applies whether deployed on heroku (e.g. [http://visiontracing-<suffix>.herokuapp.com](http://visiontracing-<suffix>.herokuapp.com)
at http://visiontracing-\<suffix\>.herokuapp.com) or docker-compose (e.g. at http://localhost:5000):

- Observe queues, workers, and jobs in the RQ-dashboard at the '/rq' endpoint.

- Tail logs to observe stdout from web and worker processes.

- Visit the top level '/' endpoint  to enqueue a message. Refresh several times in rapid succession, then watch the RQ-dashboard and/or logs to confirm the queue is being drained by the workers.

## TODO

This currently doesn't do anything useful. Next steps:
- Update web app to allow the user to upload a video file. See https://github.com/MLDSAI/ContactTracingAI.com/blob/master/app/views/video.py#L84 for an example of how to do this.
- Refactor code for detection, tracking, and subject re-identification in the `visiontracing` directory into something that can be run in a worker.
- Display the results to the user
