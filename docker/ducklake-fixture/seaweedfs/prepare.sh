#!/bin/sh
# One-shot setup for the ducklake-s3 service, run from the official busybox
# image before SeaweedFS starts. Same pattern as casmdb/deploy/seaweedfs.
# dhi.io/seaweedfs has no shell, so this:
#
#   1. writes the S3 identity file from the environment into /config;
#   2. copies a static busybox into /tools for the healthcheck and the
#      bucket-init job (both run in the SeaweedFS image);
#   3. hands /data to the image's non-root user (65532), which also fixes
#      volumes created by the earlier root-run image.
#
# The fixture's buckets are private: only the admin key pair has access,
# as with the MinIO setup this replaced (it set no anonymous policy).
#
#   S3_ACCESS_KEY_ID / S3_SECRET_ACCESS_KEY  admin key pair (read/write/admin)
set -eu

: "${S3_ACCESS_KEY_ID:?S3_ACCESS_KEY_ID must be set}"
: "${S3_SECRET_ACCESS_KEY:?S3_SECRET_ACCESS_KEY must be set}"
SEAWEEDFS_UID=65532

umask 077
cat > /config/s3.json <<JSON
{
  "identities": [
    {
      "name": "admin",
      "credentials": [{"accessKey": "${S3_ACCESS_KEY_ID}", "secretKey": "${S3_SECRET_ACCESS_KEY}"}],
      "actions": ["Admin", "Read", "List", "Tagging", "Write"]
    }
  ]
}
JSON
chown "$SEAWEEDFS_UID:$SEAWEEDFS_UID" /config/s3.json

umask 022
cp /bin/busybox /tools/busybox
/tools/busybox --install -s /tools

# SeaweedFS keeps everything in /data/store, not at the volume's top level:
# when the SeaweedFS container first mounts a new volume, Podman resets the
# owner of its top-level folder to match the image's root-owned /data (even
# with nocopy), after this job has run. A subfolder keeps its owner.
mkdir -p /data/store
# Volumes from before /data/store kept SeaweedFS's files at the top level;
# move them in (the master, volume and filer stores are found relative to it).
for entry in /data/* /data/.[!.]* /data/..?*; do
  [ -e "$entry" ] || continue
  [ "$entry" = /data/store ] && continue
  mv "$entry" /data/store/
done
chown -R "$SEAWEEDFS_UID:$SEAWEEDFS_UID" /data/store
echo "SeaweedFS config, tools and data volume ready."
