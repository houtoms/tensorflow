
# This is a convenience wrapper for checking out a working branch from a
# *-devel branch according to our fancy merge workflow.
# This should be used after pulling an updated *-devel branch.

set -e # Bail on any failure
function cleanup {
  git checkout $current_branch
}
trap cleanup ERR SIGINT

current_branch=`git rev-parse --abbrev-ref HEAD`
if [[ $1 =~ ^(-h|--help)$ ]]; then
    echo "Usage: begin_working.sh [branch_name]"
    exit
elif ! [ -z "$1" ]; then
    working_branch=$1
else
    working_branch="working"
fi
if [[ $current_branch != *"-devel" ]]; then
    echo "ERROR: You are not on a *-devel branch"
    exit -1
fi
git checkout HEAD^2 || {
retcode=$?
last_merge_hash=`git log | grep -C 1 -m 1 Merge | head -n 1 | cut -d' ' -f2`
echo "--------------------------------------------------------------------------------"
echo "Oops! It looks like there are commits in $current_branch that should not be there."
echo "They should be rebased or cherry-picked back to the last merge point at:"
echo
git log | grep -A 4 -m 1 Merge
echo "--------------------------------------------------------------------------------"

read -p "Checkout this last merge commit? [y/N] " response
if [[ $response =~ ^([yY][eE][sS]|[yY])$ ]]
then
    git checkout $last_merge_hash
    git checkout HEAD^2
else
    exit $retcode
fi
}
git checkout -b $working_branch || {
retcode=$?
echo "--------------------------------------------------------------------------------"
echo "You already have a local $working_branch branch; you can either:"
echo "  [d] Delete it and start a new one from the current HEAD,"
echo "  [r] Rebase it onto the current HEAD, or"
echo "  [A] Abort"
echo "--------------------------------------------------------------------------------"
read -p "Delete, rebase or abort? [d/r/A] " response
if [[ $response =~ ^([dD][eE][lL][eE][tT][eE]|[dD])$ ]]; then
  git branch   -d $working_branch
  git checkout -b $working_branch
elif [[ $response =~ ^([rR][eE][bB][aA][sS][eE]|[rR])$ ]]; then
  head_hash=`git rev-parse --verify HEAD`
  git checkout $working_branch
  git rebase $working_branch --onto $head_hash
else
  cleanup
  exit $retcode
fi
}
