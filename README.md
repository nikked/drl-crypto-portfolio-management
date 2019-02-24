
## TODO

* feature: a small test dataset to run through full pipeline
    * can be run with unittests to rapidly check scenarios
* refactor main.py
* write a small unit test that is run on a precommit hook


### Git precommit hooks
Make sure you are using the same git hooks as defined in .githooks!

Please run:
`git config core.hooksPath .githooks`
`chmod -R  777 .githooks`

This ensures that certain test procedures are ran before a commit is allowed