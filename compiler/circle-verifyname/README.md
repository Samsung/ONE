# circle-verifyname

_circle-verifyname_ allows users to check name is not empty and it's unique for
all nodes in a circle model.

## Usage

Provide _circle_ file as a parameter to check uniqueness.

```
$ circle_verifyname circlefile.circle
```

Result for all names exist and are unique in the file
```
[ RUN       ] Check circlefile.circle
[      PASS ] Check circlefile.circle
```

Result for empty or duplicate names in the file
```
[ RUN       ] Check circlefile.circle
[      FAIL ] Check circlefile.circle
```
