#!/tools/sh
# Create S3_BUCKET if it doesn't exist yet. Safe to re-run. Runs in the
# SeaweedFS image with the busybox tools that prepare.sh put in /tools.
set -eu
PATH="/tools:$PATH"

MASTER="${SEAWEEDFS_MASTER:-ducklake-s3:9333}"
S3_BUCKET="${S3_BUCKET:-casmsocial-ducklake}"

if echo "s3.bucket.list" | weed shell -master="$MASTER" 2>/dev/null | grep -Eq "^[[:space:]]*${S3_BUCKET}([[:space:]]|$)"; then
  echo "Bucket ${S3_BUCKET} already exists."
else
  echo "s3.bucket.create -name ${S3_BUCKET}" | weed shell -master="$MASTER"
  echo "Bucket ${S3_BUCKET} created."
fi
