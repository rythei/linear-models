# Updating the Jupyter book

From the parent directory containing linear-models/ run

`jupyter-book build --all linear-models/`

Next, push the updates to the github repo, e.g.

`git add ./*`
`git commit -m "update stuff"`
`git push`

Finally, add the updates to the gh-pages branch for publishing

`ghp-import -n -p -f _build/html`

jupyter-book toc migrate linear-models/_toc.yml -o linear-models/_toc.yml
