# GTSAM Notes
- A factor refers to nodes in the graph. But a stock factor cannot refer to some function of nodes in the graph. You'd have to write a new factor to do that, or potentially use an `ExpressionFactor` (?), which is not available in the python API.
 - I need to this apply a shared & optimizable camera_from_body offset orientation.
  - There's no obvious way to do this with the python API.  There is a field for this in some of the projection factors, but it only accepts a fixed constant offset!
   - `ReferenceFrameFactor` seems exactly what I need, but is not in python api.
   - Mayve `BetweenFactor` is applicable: create a shared bias node and _2_ nodes for each new pose.
- The names of nodes & factors is not straightforward. It looked like I needed `GenericProjectionFactor`, but actually `GeneralSFMFactor2Cal3_S2` was needed. So many factors are needed to be provided because, again, of the inability to create nodes recursively with expressions.
