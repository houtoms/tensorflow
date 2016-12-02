
# This is a convenience wrapper for the emulation of "git merge -s theirs"
# described here: http://stackoverflow.com/a/27338013/7228843
# This should be used INSTEAD OF COMMITTING DIRECTLY TO *-DEVEL
# to merge a local working branch back into a *-devel branch before pushing
# the updated *-devel branch.

set -e # Bail on any failure

if [ -z "$1" ] || [[ $1 =~ ^(-h|--help)$ ]]; then
  echo "Usage: fancy_merge.sh destination_branch
 E.g., fancy_merge.sh 16.12-devel"
  exit -1
fi

current_branch=`git rev-parse --abbrev-ref HEAD`
working_branch=$current_branch
if [[ $current_branch == *"-devel" ]]; then
    echo "ERROR: You are on a *-devel branch not a working branch"
    exit -1
fi
destination_branch=$1
if [[ $destination_branch != *"-devel" ]]; then
    echo "ERROR: Destination is not a *-devel branch"
    exit -1
fi
git checkout $destination_branch
# Do a merge commit. The content of this commit does not matter,
# so use a strategy that never fails.
# Note: This advances $destination_branch.
git merge -s ours $working_branch || {
  retcode=$?
  echo "Aborting fancy merge"
  git reset --merge
  git checkout $current_branch
  exit $retcode
}
# Change working tree and index to desired content.
# --detach ensures $working_branch will not move when doing the reset in the next step.
git checkout --detach $working_branch
# Move HEAD to $destination_branch without changing contents of working tree and index.
git reset --soft $destination_branch
# 'attach' HEAD to $destination_branch.
# This ensures $destination_branch will move when doing 'commit --amend'.
git checkout $destination_branch
# Change content of merge commit to current index (i.e. content of $working_branch).
git commit --amend -C HEAD
echo "Fancy merge complete. You can now \"git push origin HEAD\" or
 \"git checkout $working_branch\" and continue working."
