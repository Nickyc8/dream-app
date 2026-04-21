export async function POST(request: Request) {
  try {
    const body = await request.json();
    const dreamText = body.dreamText;

    if (!dreamText || !dreamText.trim()) {
      return Response.json(
        { error: "Dream text is required." },
        { status: 400 }
      );
    }

    return Response.json({
      emotion: "Fear",
      confidence: 82,
      cluster: 3,
    });
  } catch {
    return Response.json(
      { error: "Something went wrong while analyzing the dream." },
      { status: 500 }
    );
  }
}