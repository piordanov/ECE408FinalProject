# ECE408FinalProject
an optimization of Game of Life through GPU CUDA

Our set up steps:
to load cuda compiler
module load cuda-toolkit

set up rules:
git clone https://github.com/piordanov/ECE408FinalProject.git

to push to master:
git add <files>
git commit
//you will then have to write some message, and to save you have to type :wq
git push

At this point, there is a possibility of conflicts, if someone else has pushed changes that you have not pulled yet. Depending on the conflicts, "git merge" or manually merging changes may be necessary.

To override local commits: from http://stackoverflow.com/questions/1125968/force-git-to-overwrite-local-files-on-pull
git fetch --all
git reset --hard origin/master

