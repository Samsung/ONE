# tf2circle-ui-check

tf2circle-ui-check makes it easy to check what ``tf2circle`` shows for selected TensorFlow testcases.

## HOW TO USE

First of all, create "test.lst" file and add tests of interest. Here is an example of "test.lst"
```
Add(NET_0000)
Add(NET_0001)
```

Run "nncc configure". You may find the below messages if ``tf2circle-ui-check`` is configured properly:
```
-- Configure TF2CIRCLE-UI-CHECK
-- Build tf2circle-ui-check: TRUE
-- Configure TF2CIRCLE-UI-CHECK - Done
```

Finally, build ``tf2circle_ui_check`` target and see what happens!
If CMake uses "make" as a generator, you may build ``tf2circle_ui_check`` target via running ``./nncc build tf2circle_ui_check``.
